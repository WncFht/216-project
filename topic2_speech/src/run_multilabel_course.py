#!/usr/bin/env python3
"""Course-method multi-label voiced consonant detector.

This script addresses the upgraded requirement:
- detect all voiced consonants covered by the expanded Speech Commands subset;
- reduce single-word fitting by using multiple words per consonant where possible;
- implement a MATLAB-style course baseline based on FFT/STFT spectral templates;
- include an optional CUDA CNN multi-label detector for comparison.
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import get_window
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    torch = None; nn = None; DataLoader = None; TensorDataset = None; TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

RNG = 216
# Voiced consonants covered by the selected real words.  /zh/ is represented by
# "visual"; English /ŋ/ and /ð/ are not covered reliably by Speech Commands words.
LABELS = ["b", "d", "g", "j", "l", "m", "n", "r", "v", "w", "z", "zh"]
L2I = {c:i for i,c in enumerate(LABELS)}
# Manually curated ARPABET-like voiced-consonant presence labels for selected
# Speech Commands words.  Labels describe pronunciation, not spelling.
WORD_LABELS: Dict[str, List[str]] = {
    "backward": ["b", "w", "r", "d"],
    "bed": ["b", "d"],
    "bird": ["b", "r", "d"],
    "dog": ["d", "g"],
    "down": ["d", "n"],
    "go": ["g"],
    "learn": ["l", "r", "n"],
    "left": ["l"],
    "marvin": ["m", "r", "v", "n"],
    "nine": ["n"],
    "no": ["n"],
    "right": ["r"],
    "visual": ["v", "zh", "l"],
    "wow": ["w"],
    "yes": ["j"],
    "zero": ["z", "r"],
    "five": ["v"],
    "seven": ["v", "n"],
    "one": ["w", "n"],
    # Negative / mostly unvoiced/vowel controls.
    "stop": [], "up": [], "tree": ["r"], "three": ["r"], "off": [], "six": [],
}
WORDS = list(WORD_LABELS.keys())


def seed_all(seed=RNG):
    random.seed(seed); np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_wav(path: Path, target_sr=16000):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1: x = x.mean(axis=1)
    if np.max(np.abs(x)) > 0: x = x / np.max(np.abs(x))
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr//g, sr//g).astype(np.float32); sr = target_sr
    if len(x) < target_sr: x = np.pad(x, (0, target_sr-len(x)))
    else: x = x[:target_sr]
    return sr, x.astype(np.float32)


def hz_to_mel(f): return 2595*np.log10(1+np.asarray(f)/700)
def mel_to_hz(m): return 700*(10**(np.asarray(m)/2595)-1)

def mel_filterbank(sr, n_fft, n_mels=40, fmin=50, fmax=None):
    if fmax is None: fmax=sr/2
    mels=np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels+2)
    hz=mel_to_hz(mels); bins=np.floor((n_fft+1)*hz/sr).astype(int)
    fb=np.zeros((n_mels,n_fft//2+1), dtype=np.float32)
    for m in range(1,n_mels+1):
        l,c,r=bins[m-1],bins[m],bins[m+1]; c=max(c,l+1); r=max(r,c+1)
        for k in range(l, min(c, fb.shape[1])): fb[m-1,k]=(k-l)/(c-l)
        for k in range(c, min(r, fb.shape[1])): fb[m-1,k]=(r-k)/(r-c)
    return fb


def frame_signal(x, sr, frame_ms=25, hop_ms=10):
    fl=int(sr*frame_ms/1000); hop=int(sr*hop_ms/1000)
    n=1+(len(x)-fl)//hop
    frames=np.lib.stride_tricks.as_strided(x, shape=(n,fl), strides=(x.strides[0]*hop,x.strides[0])).copy()
    frames *= get_window('hamming', fl, fftbins=True)
    return frames


def reps(x, sr):
    x = np.append(x[0], x[1:] - 0.97*x[:-1]).astype(np.float32)
    frames=frame_signal(x, sr)
    n_fft=512
    mag=np.abs(np.fft.rfft(frames, n=n_fft))+1e-12
    power=(mag**2)/n_fft
    fb=mel_filterbank(sr,n_fft,40)
    logmel=np.log(np.maximum(power @ fb.T, 1e-12)).astype(np.float32) # T x M
    mfcc=dct(logmel, type=2, axis=1, norm='ortho')[:,:13].astype(np.float32)
    delta=np.gradient(mfcc, axis=0).astype(np.float32)
    # Course FFT band energies: coarse spectrum analyzer features.
    freqs=np.fft.rfftfreq(n_fft, 1/sr)
    band_edges=np.array([0,250,500,750,1000,1500,2000,3000,4000,5500,8000], dtype=float)
    bands=[]
    for lo,hi in zip(band_edges[:-1], band_edges[1:]):
        mask=(freqs>=lo)&(freqs<hi)
        bands.append(np.log(np.mean(mag[:,mask]**2)+1e-12))
    band_vec=np.asarray(bands, dtype=np.float32)
    # Full STFT template vector is still course-based; no learned representation.
    stft_vec=logmel.reshape(-1)
    stat=np.concatenate([mfcc.mean(0), mfcc.std(0), delta.mean(0), delta.std(0), band_vec]).astype(np.float32)
    return band_vec, stat, stft_vec, logmel


def collect(data_roots: List[Path], max_per_word=650):
    rows=[]
    rng=random.Random(RNG)
    seen=set()
    for root in data_roots:
        for word in WORDS:
            d=root/word
            if not d.exists(): continue
            files=sorted(d.glob('*.wav'))
            rng.shuffle(files)
            files=files[:max_per_word]
            for p in files:
                key=str(p.resolve())
                if key in seen: continue
                seen.add(key)
                speaker=p.name.split('_nohash_')[0] if '_nohash_' in p.name else p.stem.split('_')[0]
                y=np.zeros(len(LABELS), dtype=np.float32)
                for lab in WORD_LABELS[word]: y[L2I[lab]]=1
                rows.append((p,word,speaker,y))
    rng.shuffle(rows)
    return rows


def build_dataset(rows, cache):
    if cache.exists(): return dict(np.load(cache, allow_pickle=True))
    X_band=[]; X_stat=[]; X_stft=[]; X_cnn=[]; Y=[]; words=[]; speakers=[]; paths=[]
    for p,w,s,y in rows:
        sr,x=load_wav(p)
        a,b,c,d=reps(x,sr)
        X_band.append(a); X_stat.append(b); X_stft.append(c); X_cnn.append(d); Y.append(y)
        words.append(w); speakers.append(s); paths.append(str(p))
    data={
        'X_band':np.vstack(X_band), 'X_stat':np.vstack(X_stat), 'X_stft':np.vstack(X_stft), 'X_cnn':np.stack(X_cnn),
        'Y':np.vstack(Y).astype(np.float32), 'words':np.asarray(words), 'speakers':np.asarray(speakers), 'paths':np.asarray(paths)
    }
    np.savez_compressed(cache, **data)
    return data


def split(Y, speakers):
    groups=np.asarray(speakers)
    strat=(Y @ (np.arange(Y.shape[1])+1)).astype(int)  # rough only; GroupShuffle ignores exact stratification
    idx=np.arange(len(Y))
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=RNG).split(idx, strat, groups))
    return tr,te


def choose_threshold(y_true, scores):
    best_t=0.0; best_f=-1
    for t in np.quantile(scores, np.linspace(0.02,0.98,97)):
        pred=(scores>=t).astype(int)
        f=f1_score(y_true, pred, zero_division=0)
        if f>best_f: best_f=f; best_t=float(t)
    return best_t


def template_detector(Xtr,Ytr,Xte):
    scaler=StandardScaler(); Ztr=scaler.fit_transform(Xtr); Zte=scaler.transform(Xte)
    scores=np.zeros((len(Zte),Ytr.shape[1]), dtype=np.float32); thresholds=[]
    model=[]
    train_scores=np.zeros((len(Ztr),Ytr.shape[1]), dtype=np.float32)
    for j in range(Ytr.shape[1]):
        pos=Ztr[Ytr[:,j]==1]; neg=Ztr[Ytr[:,j]==0]
        mu_p=pos.mean(0); mu_n=neg.mean(0)
        def score(Z):
            cp=(Z@mu_p)/((np.linalg.norm(Z,axis=1)+1e-12)*(np.linalg.norm(mu_p)+1e-12))
            cn=(Z@mu_n)/((np.linalg.norm(Z,axis=1)+1e-12)*(np.linalg.norm(mu_n)+1e-12))
            return cp-cn
        s_tr=score(Ztr); t=choose_threshold(Ytr[:,j].astype(int), s_tr)
        train_scores[:,j]=s_tr; scores[:,j]=score(Zte); thresholds.append(t); model.append((mu_p,mu_n))
    pred=(scores>=np.asarray(thresholds)[None,:]).astype(int)
    return pred, scores, np.asarray(thresholds), scaler, model


def metrics(Y, P, name):
    rows=[]
    p,r,f,s=precision_recall_fscore_support(Y, P, average=None, zero_division=0)
    for i,lab in enumerate(LABELS): rows.append({'method':name,'label':lab,'precision':p[i],'recall':r[i],'f1':f[i],'support':int(s[i])})
    return rows


class MultiCnn(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),
            nn.Linear(96,96),nn.ReLU(),nn.Dropout(.25),nn.Linear(96,n))
    def forward(self,x): return self.net(x)


def train_cnn(Xtr,Ytr,Xte,device,epochs=14):
    mean=float(Xtr.mean()); std=float(Xtr.std()+1e-6)
    Xtr=((Xtr-mean)/std)[:,None,:,:].astype(np.float32); Xte=((Xte-mean)/std)[:,None,:,:].astype(np.float32)
    ds=TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr.astype(np.float32)))
    dl=DataLoader(ds,batch_size=192,shuffle=True)
    model=MultiCnn(len(LABELS)).to(device)
    pos=Ytr.sum(0); neg=len(Ytr)-pos; pos_weight=np.clip(neg/np.maximum(pos,1),1,10)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    hist=[]
    for ep in range(1,epochs+1):
        model.train(); loss_sum=0; n=0
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True)
            loss=crit(model(xb),yb); loss.backward(); opt.step()
            loss_sum+=float(loss.item())*len(yb); n+=len(yb)
        hist.append({'epoch':ep,'loss':loss_sum/n})
    def predict_scores(X):
        scores=[]
        model.eval()
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=512):
                scores.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        return np.vstack(scores)
    train_scores = predict_scores(Xtr)
    scores = predict_scores(Xte)
    return model,train_scores,scores,mean,std,pd.DataFrame(hist)


def plot_label_coverage(df,out):
    pivot=df.groupby('word')[LABELS].max().loc[[w for w in WORDS if w in df.word.unique()]]
    plt.figure(figsize=(10,8)); sns.heatmap(pivot, cmap='Blues', cbar=False, linewidths=.5, linecolor='white')
    plt.title('Voiced-consonant presence labels by word'); plt.xlabel('voiced consonant'); plt.ylabel('word')
    plt.tight_layout(); plt.savefig(out,dpi=220); plt.close()


def plot_metrics(mdf,out):
    plt.figure(figsize=(11,5.8)); sns.barplot(data=mdf,x='label',y='f1',hue='method')
    plt.ylim(0,1); plt.title('Per-consonant detection F1'); plt.ylabel('F1'); plt.xlabel('voiced consonant')
    plt.grid(axis='y',alpha=.25); plt.tight_layout(); plt.savefig(out,dpi=220); plt.close()


def plot_template_grid(Xcnn,Y,out):
    fig,axes=plt.subplots(3,4,figsize=(13,8),constrained_layout=True)
    for ax,lab in zip(axes.ravel(),LABELS):
        i=L2I[lab]; avg=Xcnn[Y[:,i]==1].mean(0).T
        im=ax.imshow(avg,aspect='auto',origin='lower',cmap='magma')
        ax.set_title(f'/{lab}/ avg log-mel'); ax.set_xlabel('frame'); ax.set_ylabel('mel bin')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.75)
    fig.savefig(out,dpi=220); plt.close(fig)


def plot_course_score(scores,Y,label,out):
    i=L2I[label]
    plt.figure(figsize=(7,4.5))
    sns.kdeplot(scores[Y[:,i]==1,i], label=f'contains /{label}/', fill=True, alpha=.35)
    sns.kdeplot(scores[Y[:,i]==0,i], label=f'not /{label}/', fill=True, alpha=.35)
    plt.title(f'Course spectral-template score distribution for /{label}/')
    plt.xlabel('cosine(pos template) - cosine(neg template)'); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=220); plt.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-roots', nargs='+', type=Path, default=[Path('data/raw/speech_commands_v0.02'),Path('data/raw/mini_speech_commands')]); ap.add_argument('--out-dir',type=Path,default=Path('build/multilabel_course')); ap.add_argument('--fig-dir',type=Path,default=Path('figures/multilabel_course')); ap.add_argument('--max-per-word',type=int,default=650); ap.add_argument('--epochs',type=int,default=14)
    args=ap.parse_args(); seed_all(); args.out_dir.mkdir(parents=True,exist_ok=True); args.fig_dir.mkdir(parents=True,exist_ok=True)
    rows=collect(args.data_roots,args.max_per_word)
    data=build_dataset(rows,args.out_dir/'features_v1.npz')
    Y=data['Y'].astype(int); tr,te=split(Y,data['speakers'])
    Pb,Sb,Tb,_,_=template_detector(data['X_band'][tr],Y[tr],data['X_band'][te])
    Ps,Ss,Ts,_,_=template_detector(data['X_stft'][tr],Y[tr],data['X_stft'][te])
    device='cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
    model,Scnn_tr,Scnn,mean,std,hist=train_cnn(data['X_cnn'][tr],Y[tr].astype(np.float32),data['X_cnn'][te],device,args.epochs)
    cnn_thresholds=np.asarray([choose_threshold(Y[tr,j].astype(int), Scnn_tr[:,j]) for j in range(len(LABELS))])
    Pcnn=(Scnn>=cnn_thresholds[None,:]).astype(int)
    metrics_df=pd.DataFrame(metrics(Y[te],Pb,'course_fft_band')+metrics(Y[te],Ps,'course_stft_template')+metrics(Y[te],Pcnn,'cnn_logmel_multilabel'))
    summary=metrics_df.groupby('method').agg(macro_f1=('f1','mean'), weighted_f1=('f1',lambda s: np.average(s,weights=metrics_df.loc[s.index,'support']))).reset_index()
    metrics_df.to_csv(args.out_dir/'per_label_metrics.csv',index=False); summary.to_csv(args.out_dir/'summary.csv',index=False); hist.to_csv(args.out_dir/'cnn_history.csv',index=False)
    index=pd.DataFrame({'path':data['paths'],'word':data['words'],'speaker':data['speakers'],'split':['train' if i in set(tr) else 'test' for i in range(len(Y))]})
    for i,lab in enumerate(LABELS): index[lab]=Y[:,i]
    index.to_csv(args.out_dir/'dataset_index.csv',index=False)
    pd.DataFrame({'label':LABELS,'course_fft_threshold':Tb,'course_stft_threshold':Ts,'cnn_threshold':cnn_thresholds}).to_csv(args.out_dir/'thresholds.csv',index=False)
    config={'labels':LABELS,'word_labels':WORD_LABELS,'n_files':int(len(Y)),'n_train':int(len(tr)),'n_test':int(len(te)),'n_speakers':int(len(set(data['speakers']))),'train_speakers':int(len(set(data['speakers'][tr]))),'test_speakers':int(len(set(data['speakers'][te]))),'device':device,'torch':None if torch is None else torch.__version__,'cuda_device':None if torch is None or not torch.cuda.is_available() else torch.cuda.get_device_name(0)}
    (args.out_dir/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    plot_label_coverage(index,args.fig_dir/'label_coverage.png'); plot_metrics(metrics_df,args.fig_dir/'per_label_f1.png'); plot_template_grid(data['X_cnn'][tr],Y[tr],args.fig_dir/'all_voiced_templates.png'); plot_course_score(Ss,Y[te],'b',args.fig_dir/'score_distribution_b.png'); plot_course_score(Ss,Y[te],'v',args.fig_dir/'score_distribution_v.png')
    print(json.dumps(config,ensure_ascii=False,indent=2)); print('\nSummary'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}')); print('\nPer label'); print(metrics_df.to_string(index=False,float_format=lambda x:f'{x:.3f}'))

if __name__=='__main__': main()
