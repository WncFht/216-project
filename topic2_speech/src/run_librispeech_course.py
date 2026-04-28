#!/usr/bin/env python3
"""LibriSpeech + CMUdict voiced-consonant presence experiment.

This replaces short keyword-only data with read-sentence audio.  Labels are
utterance-level voiced-consonant presence labels derived from transcripts through
CMUdict/ARPABET.  The course baseline is a MATLAB-style FFT/STFT spectral
template detector; CNN is included only as a comparison.
"""
from __future__ import annotations
import argparse, json, math, re, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly, stft
from scipy.fft import dct
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, f1_score
import soundfile as sf
import pronouncing

try:
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as exc:
    torch=None; nn=None; TensorDataset=None; DataLoader=None; TORCH_ERR=exc
else:
    TORCH_ERR=None

RNG=216
LABELS=['b','d','dh','g','jh','l','m','n','ng','r','v','w','y','z','zh']
ARPABET_TO_LABEL={'B':'b','D':'d','DH':'dh','G':'g','JH':'jh','L':'l','M':'m','N':'n','NG':'ng','R':'r','V':'v','W':'w','Y':'y','Z':'z','ZH':'zh'}
L2I={l:i for i,l in enumerate(LABELS)}
WORD_RE=re.compile(r"[A-Z']+")
EXAMPLE_TARGETS=[
    ('Example 1', ('b','dh','v')),
    ('Example 2', ('ng','z')),
    ('Example 3', ('jh','r')),
]


def seed_all():
    random.seed(RNG); np.random.seed(RNG)
    if torch is not None:
        torch.manual_seed(RNG)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(RNG)

def read_audio(path:Path,target_sr=16000,max_sec=6.0):
    x,sr=sf.read(path, dtype='float32')
    if x.ndim>1: x=x.mean(axis=1)
    if sr!=target_sr:
        g=math.gcd(sr,target_sr); x=resample_poly(x,target_sr//g,sr//g).astype(np.float32); sr=target_sr
    n=int(max_sec*target_sr)
    if len(x)>n: x=x[:n]
    if len(x)<n: x=np.pad(x,(0,n-len(x)))
    mx=np.max(np.abs(x));
    if mx>0: x=x/mx
    return sr,x.astype(np.float32)

def labels_from_text(text:str):
    y=np.zeros(len(LABELS),dtype=np.float32); missing=[]; phones=[]
    for word in WORD_RE.findall(text.upper()):
        prons=pronouncing.phones_for_word(word.lower().replace("'",''))
        if not prons:
            missing.append(word); continue
        # choose first pronunciation, strip stress digits
        for ph in prons[0].split():
            base=re.sub(r'\d','',ph)
            phones.append(base)
            lab=ARPABET_TO_LABEL.get(base)
            if lab: y[L2I[lab]]=1
    return y, phones, missing

def collect(root:Path,max_utts:int):
    rows=[]
    for trans in sorted(root.glob('**/*.trans.txt')):
        for line in trans.read_text().splitlines():
            if not line.strip(): continue
            utt,*words=line.split(maxsplit=1); text=words[0] if words else ''
            audio=trans.parent/f'{utt}.flac'
            if not audio.exists(): continue
            y,phones,missing=labels_from_text(text)
            if y.sum()==0: continue
            speaker=utt.split('-')[0]
            rows.append((audio,utt,text,speaker,y,phones,missing))
    random.Random(RNG).shuffle(rows)
    return rows[:max_utts]

def hz_to_mel(f): return 2595*np.log10(1+np.asarray(f)/700)
def mel_to_hz(m): return 700*(10**(np.asarray(m)/2595)-1)
def mel_fb(sr,n_fft,n_mels=48):
    hz=mel_to_hz(np.linspace(hz_to_mel(50),hz_to_mel(sr/2),n_mels+2)); bins=np.floor((n_fft+1)*hz/sr).astype(int)
    fb=np.zeros((n_mels,n_fft//2+1),np.float32)
    for m in range(1,n_mels+1):
        l,c,r=bins[m-1],bins[m],bins[m+1]; c=max(c,l+1); r=max(r,c+1)
        for k in range(l,min(c,fb.shape[1])): fb[m-1,k]=(k-l)/(c-l)
        for k in range(c,min(r,fb.shape[1])): fb[m-1,k]=(r-k)/(r-c)
    return fb

def frame(x,sr):
    fl=int(.025*sr); hop=int(.010*sr); n=1+(len(x)-fl)//hop
    fr=np.lib.stride_tricks.as_strided(x,shape=(n,fl),strides=(x.strides[0]*hop,x.strides[0])).copy()
    fr*=get_window('hamming',fl,fftbins=True); return fr

def features(x,sr):
    x=np.append(x[0],x[1:]-0.97*x[:-1]).astype(np.float32); fr=frame(x,sr); n_fft=512
    mag=np.abs(np.fft.rfft(fr,n=n_fft))+1e-12; pow=(mag**2)/n_fft; freqs=np.fft.rfftfreq(n_fft,1/sr)
    fb=mel_fb(sr,n_fft,48); logmel=np.log(np.maximum(pow@fb.T,1e-12)).astype(np.float32)
    mfcc=dct(logmel,type=2,axis=1,norm='ortho')[:,:13].astype(np.float32)
    edges=np.array([0,150,300,500,750,1000,1500,2000,3000,4000,5500,8000])
    bands=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(freqs>=lo)&(freqs<hi); bands.append(np.log(np.mean(mag[:,mask]**2)+1e-12))
    bands=np.asarray(bands,np.float32)
    # For sentence-length utterances, summarize by mean/std and high-percentile to reflect intermittent consonants.
    stat=np.concatenate([mfcc.mean(0),mfcc.std(0),np.percentile(mfcc,90,axis=0),bands]).astype(np.float32)
    stft_vec=np.concatenate([logmel.mean(0),logmel.std(0),np.percentile(logmel,90,axis=0)]).astype(np.float32)
    return bands,stat,stft_vec,logmel

def build(rows,cache):
    if cache.exists(): return dict(np.load(cache,allow_pickle=True))
    Xb=[]; Xs=[]; Xt=[]; Xc=[]; Y=[]; meta=[]
    for audio,utt,text,speaker,y,phones,missing in rows:
        sr,x=read_audio(audio); b,s,t,c=features(x,sr)
        Xb.append(b); Xs.append(s); Xt.append(t); Xc.append(c); Y.append(y)
        meta.append((str(audio),utt,text,speaker,' '.join(phones),' '.join(missing)))
    data={'X_band':np.vstack(Xb),'X_stat':np.vstack(Xs),'X_stft':np.vstack(Xt),'X_cnn':np.stack(Xc),'Y':np.vstack(Y).astype(np.float32),'meta':np.asarray(meta,dtype=object)}
    np.savez_compressed(cache,**data); return data

def split(Y,meta):
    speakers=meta[:,3]; idx=np.arange(len(Y)); tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=RNG).split(idx,Y.sum(1),speakers)); return tr,te

def choose_t(y,s):
    best=(0,.5)
    for t in np.quantile(s,np.linspace(.02,.98,97)):
        f=f1_score(y,(s>=t).astype(int),zero_division=0)
        if f>best[0]: best=(f,float(t))
    return best[1]
def template(Xtr,Ytr,Xte):
    sc=StandardScaler(); Ztr=sc.fit_transform(Xtr); Zte=sc.transform(Xte); scores=np.zeros((len(Zte),Ytr.shape[1])); th=[]
    for j in range(Ytr.shape[1]):
        pos=Ztr[Ytr[:,j]==1]; neg=Ztr[Ytr[:,j]==0]
        mp=pos.mean(0); mn=neg.mean(0)
        def s(Z): return (Z@mp)/((np.linalg.norm(Z,axis=1)+1e-12)*(np.linalg.norm(mp)+1e-12))-(Z@mn)/((np.linalg.norm(Z,axis=1)+1e-12)*(np.linalg.norm(mn)+1e-12))
        strn=s(Ztr); th.append(choose_t(Ytr[:,j].astype(int),strn)); scores[:,j]=s(Zte)
    return (scores>=np.asarray(th)[None,:]).astype(int),scores,np.asarray(th)

class Cnn(nn.Module):
    def __init__(self,n):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(96,96),nn.ReLU(),nn.Dropout(.25),nn.Linear(96,n))
    def forward(self,x): return self.net(x)
def cnn_train(Xtr,Ytr,Xte,epochs,device):
    mean=float(Xtr.mean()); std=float(Xtr.std()+1e-6); Xtr=((Xtr-mean)/std)[:,None,:,:].astype(np.float32); Xte=((Xte-mean)/std)[:,None,:,:].astype(np.float32)
    ds=TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr.astype(np.float32))); dl=DataLoader(ds,batch_size=64,shuffle=True)
    m=Cnn(len(LABELS)).to(device); pos=Ytr.sum(0); neg=len(Ytr)-pos; pw=np.clip(neg/np.maximum(pos,1),1,8)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,dtype=torch.float32,device=device)); opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4)
    hist=[]
    for ep in range(1,epochs+1):
        m.train(); tot=0;n=0
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=crit(m(xb),yb); loss.backward(); opt.step(); tot+=float(loss.item())*len(yb); n+=len(yb)
        hist.append({'epoch':ep,'loss':tot/n})
    def pred(X):
        out=[]; m.eval()
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=128): out.append(torch.sigmoid(m(xb.to(device))).cpu().numpy())
        return np.vstack(out)
    return pred(Xtr),pred(Xte),pd.DataFrame(hist)

def mrows(Y,P,name):
    p,r,f,s=precision_recall_fscore_support(Y,P,average=None,zero_division=0)
    return [{'method':name,'label':lab,'precision':p[i],'recall':r[i],'f1':f[i],'support':int(s[i])} for i,lab in enumerate(LABELS)]

def pick_examples(meta,Y,test_idx):
    chosen=[]; used=set(); test_list=list(test_idx)
    for name,wanted in EXAMPLE_TARGETS:
        cands=[i for i in test_list if i not in used and all(Y[i,L2I[w]]==1 for w in wanted)]
        if not cands: continue
        cands=sorted(cands,key=lambda i:(len(str(meta[i,2]).split()), len(str(meta[i,2]))))
        i=cands[0]; used.add(i); chosen.append((name,wanted,i))
    return chosen

def labels_str(bits):
    labs=['/'+LABELS[i]+'/' for i,v in enumerate(bits) if int(v)==1]
    return ', '.join(labs) if labs else '(none)'

def plots(meta,Y,Xc,metrics,stft_scores,stft_thresholds,test_idx,out,artifact_dir):
    out.mkdir(parents=True,exist_ok=True); artifact_dir.mkdir(parents=True,exist_ok=True)
    # label prevalence
    prev=pd.DataFrame({'label':LABELS,'count':Y.sum(0)})
    plt.figure(figsize=(9,4)); sns.barplot(data=prev,x='label',y='count'); plt.title('LibriSpeech utterance-level voiced-consonant label counts'); plt.tight_layout(); plt.savefig(out/'librispeech_label_counts.png',dpi=220); plt.close()
    plt.figure(figsize=(11,5.5)); sns.barplot(data=metrics,x='label',y='f1',hue='method'); plt.ylim(0,1); plt.title('LibriSpeech per-label F1'); plt.tight_layout(); plt.savefig(out/'librispeech_per_label_f1.png',dpi=220); plt.close()
    fig,axs=plt.subplots(3,5,figsize=(14,8),constrained_layout=True)
    for ax,lab in zip(axs.ravel(),LABELS):
        i=L2I[lab]; avg=Xc[Y[:,i]==1].mean(0).T; im=ax.imshow(avg,aspect='auto',origin='lower',cmap='magma'); ax.set_title('/'+lab+'/'); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7); fig.savefig(out/'librispeech_voiced_templates.png',dpi=220); plt.close(fig)
    # three example visualizations and course-method score analyses
    chosen=pick_examples(meta,Y,test_idx); records=[]; score_axes=None; overview_axes=None
    if chosen:
        score_fig,score_axes=plt.subplots(len(chosen),1,figsize=(12,3.1*len(chosen)),constrained_layout=True)
        if len(chosen)==1: score_axes=[score_axes]
        overview_fig,overview_axes=plt.subplots(len(chosen),2,figsize=(11,2.9*len(chosen)),constrained_layout=True)
        if len(chosen)==1: overview_axes=np.asarray([overview_axes])
    test_pos={int(g):i for i,g in enumerate(test_idx)}
    for k,(name,wanted,i) in enumerate(chosen,1):
        path=Path(meta[i,0]); sr,x=read_audio(path,max_sec=6); f,t,Z=stft(x,fs=sr,window='hann',nperseg=400,noverlap=240,nfft=512,boundary=None)
        fig,ax=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
        tt=np.arange(len(x))/sr; ax[0].plot(tt,x,lw=.6); ax[0].set_title('Example '+str(k)+': '+meta[i,2][:90]); ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('amp')
        ax[1].pcolormesh(t,f/1000,20*np.log10(np.abs(Z)+1e-6),shading='gouraud',cmap='magma',vmin=-80,vmax=0); ax[1].set_ylim(0,8); ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('kHz')
        fig.savefig(out/f'librispeech_example_{k}.png',dpi=220); plt.close(fig)
        oa=overview_axes[k-1]
        oa[0].plot(tt,x,lw=.6); oa[0].set_title(f'{name} waveform'); oa[0].set_xlabel('time (s)'); oa[0].set_ylabel('amp')
        oa[1].pcolormesh(t,f/1000,20*np.log10(np.abs(Z)+1e-6),shading='gouraud',cmap='magma',vmin=-80,vmax=0)
        oa[1].set_title(meta[i,2][:58]); oa[1].set_ylim(0,8); oa[1].set_xlabel('time (s)'); oa[1].set_ylabel('kHz')
        local=test_pos[int(i)]; margins=stft_scores[local]-stft_thresholds; pred=(margins>=0).astype(int); true_bits=Y[i].astype(int)
        colors=[]
        for truth,guess in zip(true_bits,pred):
            if truth and guess: colors.append('#2ca25f')
            elif truth and not guess: colors.append('#f59e0b')
            elif (not truth) and guess: colors.append('#ef4444')
            else: colors.append('#94a3b8')
        ax=score_axes[k-1]
        ax.bar(LABELS,margins,color=colors)
        ax.axhline(0,color='black',lw=1,ls='--')
        ax.set_ylabel('score-th')
        ax.set_title(f'{name}: {meta[i,2][:70]}')
        ax.set_ylim(min(-0.2,float(margins.min())-0.02), max(0.2,float(margins.max())+0.02))
        ax.text(0.99,0.98,
                'true: '+labels_str(true_bits)+'\n'+'pred: '+labels_str(pred),
                transform=ax.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25',fc='white',ec='0.7',alpha=0.95))
        top=np.argsort(margins)[::-1][:5]
        records.append({
            'example':name,
            'utt':meta[i,1],
            'text':meta[i,2],
            'wanted_labels':' '.join('/'+w+'/' for w in wanted),
            'true_labels':labels_str(true_bits),
            'predicted_labels':labels_str(pred),
            'top_margin_labels':'; '.join(f'/{LABELS[j]}/={margins[j]:.3f}' for j in top),
        })
    if chosen:
        overview_fig.savefig(out/'librispeech_examples_overview.png',dpi=220)
        plt.close(overview_fig)
        score_axes[-1].set_xlabel('voiced consonant label')
        score_fig.savefig(out/'librispeech_example_scores.png',dpi=220)
        plt.close(score_fig)
        pd.DataFrame(records).to_csv(artifact_dir/'example_analysis.csv',index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/raw/librispeech/LibriSpeech')); ap.add_argument('--max-utts',type=int,default=1800); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--out-dir',type=Path,default=Path('build/librispeech_course')); ap.add_argument('--fig-dir',type=Path,default=Path('figures/librispeech_course'))
    a=ap.parse_args(); seed_all(); a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=collect(a.root,a.max_utts); data=build(rows,a.out_dir/'features_v1.npz'); Y=data['Y'].astype(int); tr,te=split(Y,data['meta'])
    Pb,Sb,Tb=template(data['X_band'][tr],Y[tr],data['X_band'][te]); Ps,Ss,Ts=template(data['X_stft'][tr],Y[tr],data['X_stft'][te])
    device='cuda' if torch is not None and torch.cuda.is_available() else 'cpu'; Str,Ste,hist=cnn_train(data['X_cnn'][tr],Y[tr].astype(np.float32),data['X_cnn'][te],a.epochs,device)
    Cth=np.asarray([choose_t(Y[tr,j].astype(int),Str[:,j]) for j in range(len(LABELS))]); Pc=(Ste>=Cth[None,:]).astype(int)
    metrics=pd.DataFrame(mrows(Y[te],Pb,'course_fft_band')+mrows(Y[te],Ps,'course_stft_template')+mrows(Y[te],Pc,'cnn_logmel'))
    summary=metrics.groupby('method').agg(macro_f1=('f1','mean'),weighted_f1=('f1',lambda s: np.average(s,weights=metrics.loc[s.index,'support']))).reset_index()
    metrics.to_csv(a.out_dir/'per_label_metrics.csv',index=False); summary.to_csv(a.out_dir/'summary.csv',index=False); hist.to_csv(a.out_dir/'cnn_history.csv',index=False)
    idx=pd.DataFrame(data['meta'],columns=['path','utt','text','speaker','phones','missing_words']); idx['split']='train'; idx.loc[te,'split']='test'
    for j,l in enumerate(LABELS): idx[l]=Y[:,j]
    idx.to_csv(a.out_dir/'dataset_index.csv',index=False); pd.DataFrame({'label':LABELS,'fft_threshold':Tb,'stft_threshold':Ts,'cnn_threshold':Cth}).to_csv(a.out_dir/'thresholds.csv',index=False)
    cfg={'n_utts':int(len(Y)),'train':int(len(tr)),'test':int(len(te)),'speakers':int(len(set(data['meta'][:,3]))),'train_speakers':int(len(set(data['meta'][tr,3]))),'test_speakers':int(len(set(data['meta'][te,3]))),'labels':LABELS,'device':device,'torch':None if torch is None else torch.__version__,'cuda_device':None if torch is None or not torch.cuda.is_available() else torch.cuda.get_device_name(0)}
    (a.out_dir/'run_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    plots(data['meta'],Y,data['X_cnn'],metrics,Ss,Ts,te,a.fig_dir,a.out_dir)
    print(json.dumps(cfg,ensure_ascii=False,indent=2)); print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}')); print(metrics.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
if __name__=='__main__': main()
