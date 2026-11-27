// SA v4 for BOJ 18789 (814 - 2).
//  - 4-digit numbers tracked as classes (string up to reversal); 5-digit numbers L abcd (L<=NL) via flagged-start counts
//  - gain upper-bound table G[cell][digit] from the (small) missing set -> very early rejection of losing moves
//  - mixed proposals: uniform + candidates (moves that can create a missing number)
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64; typedef unsigned int u32; typedef unsigned short u16; typedef unsigned char u8;
static const int R=8, C=14, N=R*C, NLMAX=3;
static u64 rs[2];
static inline u64 rotl(u64 x,int k){return (x<<k)|(x>>(64-k));}
static inline u64 rnd(){u64 s0=rs[0],s1=rs[1],r=s0+s1;s1^=s0;rs[0]=rotl(s0,55)^s1^(s1<<14);rs[1]=rotl(s1,36);return r;}
static inline u32 rndn(u32 n){return (u32)(((rnd()>>32)*(u64)n)>>32);}
struct Ent{u32 mm; u16 id; u8 c0,c3;};
struct CW{int cnt; int w;};
struct Mask{u64 a,b;};
static inline Mask operator&(Mask x,Mask y){return {x.a&y.a,x.b&y.b};}
static inline Mask operator|(Mask x,Mask y){return {x.a|y.a,x.b|y.b};}
vector<int> nb[N];
vector<array<int,4>> walks;
vector<Ent> lst[N];
vector<u32> startList[N];
vector<u32> pv;
CW cls[10000]; CW cl[NLMAX][10000];
int hh[NLMAX][N]; int act[NLMAX][N]; int amax[NLMAX]; int revt[10000];
int g[N]; long long E=0; int T, NL=0;
static const int pw[4]={1000,100,10,1};
static u64 nbm[N][2];
static Mask dm[10], Dm[10];
static const Mask ALL={~0ULL,(1ULL<<48)-1};
static inline Mask nbOf(Mask s){ Mask r{0,0}; u64 m=s.a; while(m){int i=__builtin_ctzll(m); m&=m-1; r.a|=nbm[i][0]; r.b|=nbm[i][1];} m=s.b; while(m){int i=__builtin_ctzll(m); m&=m-1; r.a|=nbm[64+i][0]; r.b|=nbm[64+i][1];} return r; }
// transition buffer
static const int BUFN=1<<16; static int tbuf[BUFN]; static int ntb=0; static bool tbOver=false;
static inline void rec(int key){ if(ntb<BUFN) tbuf[ntb++]=key; else tbOver=true; }
// missing set
static int posInM[(1+NLMAX)*10000]; static vector<int> Mlist;
static inline bool keyMissing(int key){ int t=key/10000, s=key%10000; return t==0? (cls[s].w && cls[s].cnt==0) : (cl[t-1][s].w && cl[t-1][s].cnt==0); }
static inline int keyW(int key){ int t=key/10000, s=key%10000; return t==0? cls[s].w : cl[t-1][s].w; }
static void rebuildM(){ for(int k:Mlist) posInM[k]=-1; Mlist.clear(); for(int k=0;k<(1+NL)*10000;k++) if(keyMissing(k)){ posInM[k]=Mlist.size(); Mlist.push_back(k);} }
static void syncM(){ if(tbOver){ rebuildM(); return; } for(int i=0;i<ntb;i++){ int k=tbuf[i]; bool ms=keyMissing(k); int p=posInM[k]; if(ms && p<0){ posInM[k]=Mlist.size(); Mlist.push_back(k);} else if(!ms && p>=0){ int last=Mlist.back(); Mlist[p]=last; posInM[last]=p; Mlist.pop_back(); posInM[k]=-1; } } }
// gain bound table
static int Gt[N*10]; static vector<int> touched; static vector<int> cand;
static void computeG(){
    for(int idx:touched) Gt[idx]=0; touched.clear(); cand.clear();
    for(int key:Mlist){
        int tp=key/10000, s=key%10000; int t[5]; int n; int w=keyW(key);
        if(tp==0){ n=4; t[0]=s/1000; t[1]=s/100%10; t[2]=s/10%10; t[3]=s%10; }
        else { n=5; t[0]=tp; t[1]=s/1000; t[2]=s/100%10; t[3]=s/10%10; t[4]=s%10; }
        Mask Lend[6], Rst[6], NLm[6], NRm[6];
        NLm[0]=ALL; Lend[1]=dm[t[0]]; NLm[1]=nbOf(Lend[1]);
        for(int p=2;p<=n-1;p++){ Lend[p]=NLm[p-1]&dm[t[p-1]]; NLm[p]=nbOf(Lend[p]); }
        NRm[n-1]=ALL; Rst[n-2]=dm[t[n-1]]; NRm[n-2]=nbOf(Rst[n-2]);
        for(int p=n-3;p>=0;p--){ Rst[p]=NRm[p+1]&dm[t[p+1]]; NRm[p]=nbOf(Rst[p]); }
        Mask cm[10]; bool used[10]={0,0,0,0,0,0,0,0,0,0};
        for(int p=0;p<n;p++){ int b=t[p]; if(!used[b]){used[b]=true; cm[b]={0,0};} cm[b]=cm[b]|(NLm[p]&NRm[p]); }
        for(int p=0;p<n;p++) for(int q=p+2;q<n;q++) if(t[p]==t[q]){ int b=t[p]; Mask m=NLm[p]&NRm[q]&Dm[t[p+1]]; if(q-p>=3) m=m&Dm[t[q-1]]; cm[b]=cm[b]|m; }
        if(n==5 && t[0]==t[2] && t[2]==t[4]) cm[t[0]]=cm[t[0]]|(Dm[t[1]]&Dm[t[3]]);
        for(int b=0;b<10;b++) if(used[b]){ Mask m=cm[b]; m.a&=~dm[b].a; m.b&=~dm[b].b;
            u64 x=m.a; while(x){int c=__builtin_ctzll(x); x&=x-1; int idx=c*10+b; if(!Gt[idx]){touched.push_back(idx);} Gt[idx]+=w; }
            x=m.b; while(x){int c=64+__builtin_ctzll(x); x&=x-1; int idx=c*10+b; if(!Gt[idx]){touched.push_back(idx);} Gt[idx]+=w; } }
    }
    cand=touched;
}
static inline void toggle(int u,int l,int sgn){
    CW* A=cl[l]; long long de=0; int kb=(l+1)*10000;
    for(u32 e: startList[u]){ u32 x=pv[e>>1]; u32 s=(e&1)?(x>>16):(x&0xffff);
        if(sgn>0){ if(A[s].cnt++==0){ int w=A[s].w; if(w){de-=w; rec(kb+s);} } } else { if(--A[s].cnt==0){ int w=A[s].w; if(w){de+=w; rec(kb+s);} } } }
    E+=de;
}
// returns false if early-rejected (state fully restored), true if move applied.
static inline bool tryMove(int c,int b,long long lossLimit){
    int a=g[c]; int d=b-a; long long E0=E;
    bool offA = (a>=1 && a<=NL);
    if(offA){ int l=a-1; int nn=nb[c].size(); CW* A=cl[l]; int kb=(l+1)*10000;
        for(int k=0;k<nn;k++){ int u=nb[c][k]; if(--hh[l][u]==0 && act[l][u]){
            act[l][u]=0;
            long long de=0; long long lim=lossLimit-(E-E0); const u32* sl=startList[u].data(); int n=startList[u].size(); int i=0; bool bad=false;
            for(;i<n;i++){ u32 e=sl[i]; u32 x=pv[e>>1]; u32 s=(e&1)?(x>>16):(x&0xffff);
                if(--A[s].cnt==0){ int w=A[s].w; if(w){ de+=w; rec(kb+s); if(de>lim){ i++; bad=true; break; } } } }
            E+=de;
            if(bad){ for(int j=0;j<i;j++){ u32 e=sl[j]; u32 x=pv[e>>1]; u32 s=(e&1)?(x>>16):(x&0xffff); A[s].cnt++; }
                act[l][u]=1; hh[l][u]++;
                for(int j=0;j<k;j++){ int uj=nb[c][j]; if(hh[l][uj]++==0 && g[uj]<=amax[l]){ act[l][uj]=1; toggle(uj,l,+1);} }
                E=E0; return false; }
        } }
    }
    {
    const Ent* base=lst[c].data(); int n=lst[c].size();
    long long de=0; long long lim=lossLimit-(E-E0);
    int i=0;
    if(NL==0){
        for(;i<n;++i){ const Ent* p=base+i; u32 x=pv[p->id]; u32 v=x&0xffff, rv=x>>16; u32 ko=v<rv?v:rv;
            if(--cls[ko].cnt==0){ int w=cls[ko].w; if(w){ de+=w; rec(ko); if(de>lim){ ++i; goto undo; } } } }
    } else {
        for(;i<n;++i){ const Ent* p=base+i; u32 x=pv[p->id]; u32 v=x&0xffff, rv=x>>16; u32 ko=v<rv?v:rv;
            if(--cls[ko].cnt==0){ int w=cls[ko].w; if(w){ de+=w; rec(ko);} }
            for(int l=0;l<NL;l++){ CW* A=cl[l];
                if(act[l][p->c0]){ if(--A[v].cnt==0){ int w=A[v].w; if(w){de+=w; rec((l+1)*10000+v);} } }
                if(act[l][p->c3]){ if(--A[rv].cnt==0){ int w=A[rv].w; if(w){de+=w; rec((l+1)*10000+rv);} } } }
            if(de>lim){ ++i; goto undo; } }
    }
    g[c]=b; for(int l=0;l<NL;l++) act[l][c]=(hh[l][c]>0 && b<=amax[l]);
    if(NL==0){
        for(int j=0;j<n;++j){ const Ent* p=base+j; u32 x=pv[p->id]; x+=(u32)(d*(int)p->mm); pv[p->id]=x; u32 v=x&0xffff, rv=x>>16; u32 kn=v<rv?v:rv;
            if(cls[kn].cnt++==0){ int w=cls[kn].w; if(w){ de-=w; rec(kn);} } }
    } else {
        for(int j=0;j<n;++j){ const Ent* p=base+j; u32 x=pv[p->id]; x+=(u32)(d*(int)p->mm); pv[p->id]=x; u32 v=x&0xffff, rv=x>>16; u32 kn=v<rv?v:rv;
            if(cls[kn].cnt++==0){ int w=cls[kn].w; if(w){ de-=w; rec(kn);} }
            for(int l=0;l<NL;l++){ CW* A=cl[l];
                if(act[l][p->c0]){ if(A[v].cnt++==0){ int w=A[v].w; if(w){de-=w; rec((l+1)*10000+v);} } }
                if(act[l][p->c3]){ if(A[rv].cnt++==0){ int w=A[rv].w; if(w){de-=w; rec((l+1)*10000+rv);} } } } }
    }
    E+=de;
    if(b>=1 && b<=NL){ int l=b-1; for(int u:nb[c]) if(hh[l][u]++==0 && g[u]<=amax[l]){ act[l][u]=1; toggle(u,l,+1);} }
    return true;
undo:
    if(NL==0){
        for(int j=0;j<i;++j){ const Ent* p=base+j; u32 x=pv[p->id]; u32 v=x&0xffff, rv=x>>16; u32 ko=v<rv?v:rv; cls[ko].cnt++; }
    } else {
        for(int j=0;j<i;++j){ const Ent* p=base+j; u32 x=pv[p->id]; u32 v=x&0xffff, rv=x>>16; u32 ko=v<rv?v:rv; cls[ko].cnt++;
            for(int l=0;l<NL;l++){ CW* A=cl[l]; if(act[l][p->c0]) A[v].cnt++; if(act[l][p->c3]) A[rv].cnt++; } }
    }
    }
    if(offA){ int l=a-1; for(int u:nb[c]) if(hh[l][u]++==0 && g[u]<=amax[l]){ act[l][u]=1; toggle(u,l,+1);} }
    E=E0;
    return false;
}
static int exactScore(const int* gg, int limit=1000000){
    u64 dmm[10][2]; memset(dmm,0,sizeof dmm); for(int i=0;i<N;i++) dmm[gg[i]][i>>6]|=1ULL<<(i&63);
    for(int n=1;n<=limit;n++){
        char buf[16]; int len=snprintf(buf,sizeof buf,"%d",n);
        u64 cur[2]={dmm[buf[0]-'0'][0],dmm[buf[0]-'0'][1]};
        for(int i=1;i<len && (cur[0]|cur[1]);i++){
            u64 nx[2]={0,0};
            for(int k=0;k<2;k++){ u64 m=cur[k]; while(m){ int b=__builtin_ctzll(m); m&=m-1; int c=k*64+b; nx[0]|=nbm[c][0]; nx[1]|=nbm[c][1]; } }
            int dg=buf[i]-'0'; cur[0]=nx[0]&dmm[dg][0]; cur[1]=nx[1]&dmm[dg][1];
        }
        if(!(cur[0]|cur[1])) return n-1;
    }
    return limit;
}
static void setT(int t){
    T=t; NL = t<10000?0: min(NLMAX,(t/10000));
    for(int k=0;k<10000;k++){cls[k].w=0; for(int l=0;l<NLMAX;l++) cl[l][k].w=0;}
    for(int s=1000;s<10000 && s<=T;s++){int k=min(s,revt[s]); cls[k].w++;}
    for(int l=0;l<NL;l++){ int L=l+1; for(int s=0;s<10000;s++) if(L*10000+s<=T) cl[l][s].w=1; amax[l]=min(9,(T-L*10000)/1000); }
}
static void rebuild(){
    memset(hh,0,sizeof hh);
    for(int k=0;k<10000;k++){cls[k].cnt=0; for(int l=0;l<NLMAX;l++) cl[l][k].cnt=0;}
    for(int c=0;c<N;c++) for(int u:nb[c]) { int dgt=g[u]; if(dgt>=1 && dgt<=NL) hh[dgt-1][c]++; }
    memset(act,0,sizeof act); for(int l=0;l<NL;l++) for(int c=0;c<N;c++) act[l][c]=(hh[l][c]>0 && g[c]<=amax[l]);
    for(size_t id=0;id<walks.size();id++){ auto&wk=walks[id]; u32 v=0,rv=0; for(int p=0;p<4;p++){v+=g[wk[p]]*pw[p]; rv+=g[wk[p]]*pw[3-p];} pv[id]=v|(rv<<16); cls[min(v,rv)].cnt++;
        for(int l=0;l<NL;l++){ if(act[l][wk[0]]) cl[l][v].cnt++; if(act[l][wk[3]]) cl[l][rv].cnt++; } }
    E=0; for(int k=0;k<10000;k++){ if(cls[k].w && !cls[k].cnt) E+=cls[k].w; for(int l=0;l<NL;l++) if(cl[l][k].w && !cl[l][k].cnt) E+=cl[l][k].w; }
    for(int x=0;x<10;x++) dm[x]={0,0}; for(int i=0;i<N;i++){ if(i<64) dm[g[i]].a|=1ULL<<i; else dm[g[i]].b|=1ULL<<(i-64); }
    for(int x=0;x<10;x++) Dm[x]=nbOf(dm[x]);
    for(int k=0;k<(1+NLMAX)*10000;k++) posInM[k]=-1; Mlist.clear(); rebuildM();
}
static void save(const char* fn,const int* gg){ FILE* o=fopen(fn,"w"); for(int r=0;r<R;r++){ for(int c=0;c<C;c++) fputc('0'+gg[r*C+c],o); fputc('\n',o);} fclose(o); }
int main(int argc,char**argv){
    if(argc<10){fprintf(stderr,"usage: T seed secsA t0 t1 secsTot step init out [qc] [mcap] [debug]\n");return 1;}
    int T0=atoi(argv[1]); u64 seed=atoll(argv[2]); double secsA=atof(argv[3]); double t0=atof(argv[4]), t1=atof(argv[5]); double secsTot=atof(argv[6]); int step=atoi(argv[7]);
    const char* initf=argv[8]; const char* outp=argv[9];
    double qc= argc>10?atof(argv[10]):0.25; int MCAP= argc>11?atoi(argv[11]):150; int debug= argc>12?atoi(argv[12]):0;
    rs[0]=seed*0x9E3779B97F4A7C15ULL+1; rs[1]=seed^0xDEADBEEFCAFEBABEULL; for(int i=0;i<20;i++) rnd();
    for(int r=0;r<R;r++)for(int c=0;c<C;c++)for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){ if(!dr&&!dc)continue; int rr=r+dr,cc=c+dc; if(rr<0||rr>=R||cc<0||cc>=C)continue; nb[r*C+c].push_back(rr*C+cc);}    
    for(int c=0;c<N;c++){ nbm[c][0]=nbm[c][1]=0; for(int u:nb[c]) nbm[c][u>>6]|=1ULL<<(u&63); }
    for(int a=0;a<N;a++)for(int b:nb[a])for(int c:nb[b])for(int d:nb[c]){ array<int,4> f={a,b,c,d}, rv={d,c,b,a}; if(f<rv) walks.push_back(f);}    
    for(int s=0;s<10000;s++){int a=s/1000,b=s/100%10,c=s/10%10,d=s%10; revt[s]=d*1000+c*100+b*10+a;}
    for(size_t id=0;id<walks.size();id++){ auto&wk=walks[id];
        startList[wk[0]].push_back(id*2); startList[wk[3]].push_back(id*2+1);
        for(int p=0;p<4;p++){ int c=wk[p]; bool first=true; for(int q=0;q<p;q++) if(wk[q]==c) first=false; if(!first) continue; u32 m=0,mr=0; for(int q=0;q<4;q++) if(wk[q]==c){m+=pw[q]; mr+=pw[3-q];} lst[c].push_back({m|(mr<<16),(u16)id,(u8)wk[0],(u8)wk[3]}); } }
    for(int c=0;c<N;c++){ auto&L=lst[c]; for(int i=(int)L.size()-1;i>0;i--){ int j=rndn(i+1); swap(L[i],L[j]); } }
    FILE* f= strcmp(initf,"-")?fopen(initf,"r"):nullptr;
    if(f){ int i=0; int ch; while(i<N && (ch=fgetc(f))!=EOF){ if(ch>='0'&&ch<='9') g[i++]=ch-'0'; } fclose(f);} else { for(int i=0;i<N;i++) g[i]=rndn(10);}    
    pv.resize(walks.size());
    setT(T0); rebuild();
    bool useG=false; if((int)Mlist.size()<=MCAP){ computeG(); useG=true; }
    fprintf(stderr,"init T=%d NL=%d E=%lld exact=%d\n",T,NL,E,exactScore(g));
    auto tstart=chrono::steady_clock::now();
    long long it=0, acc=0, early=0, viol=0; long long bestE=E; int bestg[N]; memcpy(bestg,g,sizeof g); int bestX=exactScore(g); int bestXg[N]; memcpy(bestXg,g,sizeof g);
    double temp=t0; double lastrep=0;
    const int MAXD=64; u64 thr[MAXD+2];
    auto setTemp=[&](double t){ temp=t; for(int d=0;d<=MAXD;d++){ double p=exp(-d/t); thr[d]= p>=1.0? ~0ULL : (u64)(p*18446744073709551615.0); } thr[MAXD+1]=0; };
    setTemp(t0);
    u32 qcI=(u32)(qc*4294967295.0);
    char fn[256]; snprintf(fn,sizeof fn,"%s_s%llu.txt",outp,(unsigned long long)seed);
    while(true){
        if((it&0xfff)==0){
            double el=chrono::duration<double>(chrono::steady_clock::now()-tstart).count();
            if(el>=secsTot) break;
            double fr=min(1.0,el/secsA); setTemp(t0*pow(t1/t0,fr));
            if(el-lastrep>=10.0){ lastrep=el; fprintf(stderr,"t=%.0f temp=%.3f it=%lldM acc=%.4f early=%.3f T=%d E=%lld |M|=%zu cand=%zu bestE=%lld bestX=%d\n",el,temp,it/1000000,(double)acc/(it+1),(double)early/(it+1),T,E,Mlist.size(),cand.size(),bestE,bestX); }
        }
        it++;
        int c,b;
        if(useG && !cand.empty() && (u32)(rnd()>>32)<qcI){ int idx=cand[rndn(cand.size())]; c=idx/10; b=idx%10; }
        else { c=rndn(N); b=rndn(9); if(b>=g[c]) b++; }
        int a=g[c];
        u64 r=rnd(); int th=0; while(r<thr[th+1]) th++;
        long long e0=E; ntb=0; tbOver=false;
        if(debug && useG && (it%debug)==0){
            // verify gain bound: apply fully, measure true gain of previously-missing keys
            vector<int> before=Mlist; tryMove(c,b,LLONG_MAX/4); long long gain=0; for(int k:before) if(!keyMissing(k)) gain+=keyW(k);
            if(gain>Gt[c*10+b]){ viol++; fprintf(stderr,"GAIN BOUND VIOLATION c=%d b=%d gain=%lld G=%d\n",c,b,gain,Gt[c*10+b]); }
            tryMove(c,a,LLONG_MAX/4); ntb=0; tbOver=false;
            if(E!=e0) fprintf(stderr,"REVERT MISMATCH\n");
        }
        long long lossLimit= useG ? (long long)Gt[c*10+b]+th : e0+th;
        if(!tryMove(c,b,lossLimit)){ early++; continue; }
        long long d=E-e0;
        if(d<=th){ acc++;
            syncM();
            // recompute digit masks for a and b
            dm[a]={0,0}; dm[b]={0,0}; for(int i=0;i<N;i++){ if(g[i]==a||g[i]==b){ if(i<64) dm[g[i]].a|=1ULL<<i; else dm[g[i]].b|=1ULL<<(i-64);} }
            Dm[a]=nbOf(dm[a]); Dm[b]=nbOf(dm[b]);
            if((int)Mlist.size()<=MCAP){ computeG(); useG=true; } else useG=false;
            if(E<bestE){bestE=E; memcpy(bestg,g,sizeof g);
                if(E==0){ int X=exactScore(g); double el=chrono::duration<double>(chrono::steady_clock::now()-tstart).count();
                    fprintf(stderr,"*** t=%.0f it=%lldM E=0 at T=%d exact X=%d\n",el,it/1000000,T,X);
                    if(X>bestX){bestX=X; memcpy(bestXg,g,sizeof g); save(fn,bestXg);} 
                    if(step<=0) break;
                    setT(max(T,X)+step); rebuild(); bestE=E; if((int)Mlist.size()<=MCAP){ computeG(); useG=true; } else useG=false; }
            } }
        else { tryMove(c,a,LLONG_MAX/4); }
    }
    double el=chrono::duration<double>(chrono::steady_clock::now()-tstart).count();
    long long Ecur=E; size_t ms=Mlist.size(); rebuild(); if(Ecur!=E || ms!=Mlist.size()) fprintf(stderr,"MISMATCH E %lld vs %lld, M %zu vs %zu\n",Ecur,E,ms,Mlist.size());
    fprintf(stderr,"done it=%lld (%.0f it/s) T=%d bestE=%lld bestX=%d viol=%lld\n",it,it/el,T,bestE,bestX,viol);
    char fn2[256]; snprintf(fn2,sizeof fn2,"%s_s%llu_last.txt",outp,(unsigned long long)seed); save(fn2,bestg);
    printf("%d %s\n",bestX,fn);
}
