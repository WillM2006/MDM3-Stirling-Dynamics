// This worker receives calibration and pressure packets ONLY. No sea or truth.
let weights,baseline,history,targets,taps,dt,head=-1,count=0,last=null;
function decode64(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return new Float64Array(bytes.buffer);}
self.onmessage=({data:m})=>{
 try{
  if(m.type==='init'){
   weights=decode64(m.weights64);baseline=m.baseline;targets=m.targets;taps=m.taps;dt=m.dt;
   history=new Float64Array(taps*6);head=-1;count=0;last=null;
   self.postMessage({type:'ready'});return;
  }
  if(m.type==='reset'){history.fill(0);head=-1;count=0;last=null;self.postMessage({type:'reset'});return;}
  if(m.type!=='sample'||!Array.isArray(m.pressure_Pa)||m.pressure_Pa.length!==6||
     !m.pressure_Pa.every(Number.isFinite)||!Number.isFinite(m.time_s))throw Error('Exactly six finite pressures and one timestamp required');
  if(last!==null&&Math.abs(m.time_s-last-dt)>1e-6)throw Error('Missing or out-of-order pressure sample');
  const start=performance.now();head=(head+1)%taps;count++;last=m.time_s;
  for(let c=0;c<6;c++)history[head*6+c]=m.pressure_Pa[c]-baseline[c];
  let surface=null;
  if(count>=taps){
   surface=new Float64Array(targets);
   for(let p=0;p<targets;p++){
    let value=0;
    for(let lag=0;lag<taps;lag++){
     const h=((head-lag+taps)%taps)*6,w=(p*taps+lag)*6;
     for(let c=0;c<6;c++)value+=weights[w+c]*history[h+c];
    }
    surface[p]=value;
   }
  }
  self.postMessage({type:'surface',time_s:last,count,surface,processing_ms:performance.now()-start});
 }catch(e){self.postMessage({type:'error',message:e.message});}
};
