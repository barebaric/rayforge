(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))i(s);new MutationObserver(s=>{for(const r of s)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&i(o)}).observe(document,{childList:!0,subtree:!0});function t(s){const r={};return s.integrity&&(r.integrity=s.integrity),s.referrerPolicy&&(r.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?r.credentials="include":s.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function i(s){if(s.ep)return;s.ep=!0;const r=t(s);fetch(s.href,r)}})();function _h(){return typeof window>"u"?!1:new URLSearchParams(window.location.search).has("mock")}const vh=[1,0,0,0,1,0];function kd(n,e,t){const i=t*Math.PI/180,s=Math.cos(i),r=Math.sin(i),o=e.w/2,a=e.h/2,l=[[-o,-a],[o,-a],[o,a],[-o,a],[-o,-a]],c=[];for(const[u,h]of l)c.push(n.x+u*s-h*r,n.y+u*r+h*s);return c}function zd(n,e,t,i=40){const s=[];for(let r=0;r<=i;r++){const o=r/i*Math.PI*2;s.push(n.x+e*Math.cos(o),n.y+t*Math.sin(o))}return s}function Bd(n,e){const t=[];for(let i=0;i<=10;i++){const s=i%2===0?e:e*.45,r=i/10*Math.PI*2-Math.PI/2;t.push(n.x+s*Math.cos(r),n.y+s*Math.sin(r))}return t}const Rt=[{uid:"wp-frame",name:"Outer frame",pos_mm:{x:225,y:170},size_mm:{w:220,h:140},angle_deg:0,geometry_revision:1,transform_revision:1,build:()=>kd(Rt[0].pos_mm,Rt[0].size_mm,Rt[0].angle_deg)},{uid:"wp-circle",name:"Boss circle",pos_mm:{x:60,y:90},size_mm:{w:60,h:60},angle_deg:0,geometry_revision:1,transform_revision:1,build:()=>zd(Rt[1].pos_mm,Rt[1].size_mm.w/2,Rt[1].size_mm.h/2)},{uid:"wp-star",name:"Star badge",pos_mm:{x:340,y:60},size_mm:{w:60,h:60},angle_deg:0,geometry_revision:1,transform_revision:1,build:()=>Bd(Rt[2].pos_mm,Rt[2].size_mm.w/2)}],ht=[{uid:"layer-cut",name:"Cut",visible:!0,color:"#4d9fff",is_active:!0,wcs:"G54",rotary_enabled:!1,steps:[{uid:"step-cut",step_type:"Contour",typelabel:"Contour",name:"Cut outline",enabled:!0,workpiece_uids:["wp-frame","wp-circle","wp-star"],params:{power:80,speed:3e3,passes:1,kerf:.1}}],workpieces:[]},{uid:"layer-mark",name:"Mark",visible:!0,color:"#ff9f40",is_active:!1,wcs:"G54",rotary_enabled:!1,steps:[{uid:"step-mark",step_type:"Line",typelabel:"Line fill",name:"Mark badges",enabled:!1,workpiece_uids:["wp-star"],params:{power:30,speed:5e3}}],workpieces:[]}];ht[0].workpieces=[Rt[0],Rt[1]];ht[1].workpieces=[Rt[2]];const Li={panel_orientation:"native",world_to_panel:vh,x_axis_right:!1,y_axis_down:!1,x_axis_negative:!1,y_axis_negative:!1,extents:{width_mm:400,height_mm:300},margins:{left:10,right:10,top:10,bottom:10,front:20,back:20},workarea:{x_min:10,y_min:20,x_max:390,y_max:280},workarea_origin_panel:{x:10,y:20},work_origin_panel:{x:10,y:20},active_wcs:{name:"G54",label:"WCS 1",offset_panel:{x:0,y:0}},beam_position_panel:null,soft_limits:{x_min:0,y_min:0,x_max:400,y_max:300},soft_limits_panel:{x:0,y:0,w:400,h:300},nogo_zones:[[300,200,380,200,380,280,300,280,300,200]],rotary_active:!1},gn=[{id:"mock-grbl",name:"Workbench GRBL",driver:"grbl",connected:!0,is_active:!0},{id:"mock-dummy",name:"Basement Diode",driver:"dummy",connected:!1,is_active:!1}];let At=7,Hn=!1;function ol(n){return Rt.find(e=>e.uid===n)}function al(n){for(const e of ht){const t=e.steps.find(i=>i.uid===n);if(t)return t}}const Hd=[{step_type:"Contour",typelabel:"Contour",icon:null},{step_type:"Line",typelabel:"Line fill",icon:null},{step_type:"Raster",typelabel:"Raster fill",icon:null}],Gd={Contour:{step_type:"Contour",groups:[{title:"Laser",fields:[{key:"power",type:"number",kind:"slider",label:"Power",description:null,default:80,min:0,max:100,step:1,options:null,unit:null,optional:!1},{key:"speed",type:"number",kind:"speed",label:"Speed",description:null,default:3e3,min:1,max:6e4,step:10,options:null,unit:"mm/min",optional:!1}]}]}};function Vd(n,e){const t=ht.find(i=>i.uid===n);if(!t)throw new Error(`mock: unknown layer ${n}`);e.visible!==void 0&&(t.visible=e.visible),e.name!==void 0&&(t.name=e.name),e.color!==void 0&&(t.color=e.color),e.wcs!==void 0&&(t.wcs=e.wcs),e.rotary_enabled!==void 0&&(t.rotary_enabled=e.rotary_enabled)}let cl=0;function vs(n){return cl+=1,`${n}-mock-${cl}`}const qe={status:"IDLE",connected:!0,machine_pos:{x:10,y:20,z:0},work_pos:{x:0,y:0,z:0},wco:{x:10,y:20,z:0},beam_position_panel:null,error:null,feed_rate:0,spindle_speed:0,active_wcs:"G54"},mt={active:!1,phase:"idle",progress_fraction:0,eta_seconds:null,elapsed_seconds:null},wa=new Set;function Ns(n){for(const e of wa)e(n)}function dn(){Ns({type:"machine",data:{...Ec()}})}function to(){Ns({type:"job",data:{...mt}})}function Ec(){const n=qe.beam_position_panel;return{...qe,machine_pos:qe.machine_pos&&{...qe.machine_pos},work_pos:qe.work_pos&&{...qe.work_pos},wco:qe.wco&&{...qe.wco},beam_position_panel:n&&{...n},error:qe.error&&{...qe.error}}}function en(){const n=gn.find(e=>e.is_active);if(!n||!n.connected)throw new Error("Machine is not connected");return n}const no={x:200,y:150};function Ao(n,e){no.x=n,no.y=e,qe.beam_position_panel={x:n,y:e},qe.machine_pos={x:n,y:e,z:0};const t=qe.wco?.x??0,i=qe.wco?.y??0;qe.work_pos={x:n-t,y:e-i,z:0}}const Wd={"+x":[1,0],"-x":[-1,0],"+y":[0,1],"-y":[0,-1]};let $r=null;function io(){$r!==null&&(window.clearInterval($r),$r=null)}function Xd(){io(),mt.active=!0,mt.phase="running",mt.progress_fraction=0,mt.eta_seconds=20,mt.elapsed_seconds=0,to(),$r=window.setInterval(()=>{mt.phase!=="paused"&&(mt.progress_fraction=Math.min(1,mt.progress_fraction+.02),mt.eta_seconds=Math.max(0,Math.round(20*(1-mt.progress_fraction))),mt.elapsed_seconds=Math.round(20*mt.progress_fraction),mt.progress_fraction>=1&&(io(),mt.active=!1,mt.phase="idle",qe.status="IDLE"),to())},200)}function $d(n){let e=1/0,t=1/0,i=-1/0,s=-1/0;for(let r=0;r<n.length;r+=2)e=Math.min(e,n[r]),i=Math.max(i,n[r]),t=Math.min(t,n[r+1]),s=Math.max(s,n[r+1]);return{x_min:e,y_min:t,x_max:i,y_max:s}}const xh=1800,yh=5e3;function Yd(n){let e=0;for(let t=2;t+1<n.length;t+=2)e+=Math.hypot(n[t]-n[t-2],n[t+1]-n[t-1]);return e}function qd(n){return{points:[n[0],n[1]],power:0,rate:yh,cutting:!1}}function jd(n,e){return{points:n,power:e,rate:xh,cutting:!0}}function Zd(){const n=[],e=(s,r,o)=>{o?n.push(jd(s,r)):n.push(qd(s))};e(Rt[0].build(),1,!0),e(Rt[1].build().slice(0,2),1,!1),e(Rt[1].build(),.6,!0),e(Rt[2].build().slice(0,2),.6,!1),e(Rt[2].build(),.25,!0);let t=0,i=0;for(const s of n){const r=Yd(s.points);t+=r;const o=(s.cutting?xh:yh)/60;i+=r/o}return{segments:n,estimated_seconds:Number(i.toFixed(2)),distance_mm:Number(t.toFixed(2)),generation_id:"mock-gen-1"}}function ll(){return{rev:At,layers:ht.map(n=>({...n,workpieces:n.workpieces.map(e=>({...e})),steps:n.steps.map(e=>({...e}))})),assets_count:2,stock_items:[]}}function Jd(){return{can_undo:Hn,can_redo:!1,undo_labels:Hn?["Move Star badge"]:[],redo_labels:[],dirty:Hn}}function Kd(){return{async getSystemInfo(){return{name:"rayforge",version:"0.1.0.dev0",api_version:"1",python_version:"3.13.1",machine_count:gn.length}},async getViewport(){return{...Li,world_to_panel:[...vh]}},async getDocTree(){return ll()},async getItemGeometry(n,e=.05){await ul(120);const t=ol(n);if(!t)throw new Error(`mock: unknown item ${n}`);const i=t.build();return{rev:At,polylines:[i],outline:$d(i)}},async getItemTexture(){return null},async getDocHistory(){return Jd()},async postUndo(){Hn=!1},async postRedo(){},async postNewDoc(){At++,Hn=!1},async putItemTransform(n,e){const t=ol(n);if(!t)throw new Error(`mock: unknown item ${n}`);t.pos_mm={...e.pos_mm},t.size_mm={...e.size_mm},t.angle_deg=e.angle_deg,t.transform_revision++,t.geometry_revision++,At++,Hn=!0},async putLayer(n,e){Vd(n,e),At++},async postLayer(){const e={uid:vs("layer"),name:`Layer ${ht.length+1}`,visible:!0,color:"#3ddc84",is_active:!0,wcs:null,rotary_enabled:!1,steps:[],workpieces:[]};for(const t of ht)t.is_active=!1;return ht.push(e),At++,{...e}},async deleteLayer(n){if(ht.length<=1)throw new Error("mock: cannot delete the last layer");const e=ht.findIndex(t=>t.uid===n);if(e<0)throw new Error(`mock: unknown layer ${n}`);ht.splice(e,1),At++},async activateLayer(n){if(!ht.find(t=>t.uid===n))throw new Error(`mock: unknown layer ${n}`);for(const t of ht)t.is_active=t.uid===n;At++},async reorderLayers(n){const e=new Map(ht.map(i=>[i.uid,i])),t=[];for(const i of n){const s=e.get(i);if(!s)throw new Error(`mock: unknown layer ${i}`);t.push(s)}for(const i of ht)t.includes(i)||t.push(i);ht.splice(0,ht.length,...t),At++},async getStepTypes(){return Hd.map(n=>({...n}))},async getStepSchema(n){const e=al(n),t=Gd[e?.step_type??""];if(!t)throw new Error(`mock: no schema for step ${n}`);return t},async postStep(n,e){const t=ht.find(s=>s.uid===n);if(!t)throw new Error(`mock: unknown layer ${n}`);const i={uid:vs("step"),step_type:e.step_type,typelabel:e.step_type,name:e.name??e.step_type,enabled:!0,workpiece_uids:t.workpieces.map(s=>s.uid),params:{}};return t.steps.push(i),At++,{...i,layer_uid:n}},async putStep(n,e){const t=al(n);if(!t)throw new Error(`mock: unknown step ${n}`);return e.name!==void 0&&(t.name=e.name),e.params&&(t.params={...t.params,...e.params}),At++,{...t}},async deleteStep(n){for(const e of ht){const t=e.steps.findIndex(i=>i.uid===n);if(t>=0){e.steps.splice(t,1),At++;return}}throw new Error(`mock: unknown step ${n}`)},async deleteItems(n){for(const e of ht)e.workpieces=e.workpieces.filter(t=>!n.includes(t.uid));At++},async duplicateItems(n){const e=[];for(const t of ht)for(const i of[...t.workpieces]){if(!n.includes(i.uid))continue;const s={...i,uid:vs("wp"),name:`${i.name} copy`,pos_mm:{x:i.pos_mm.x+10,y:i.pos_mm.y+10}};t.workpieces.push(s),e.push(s.uid)}return At++,{created:e}},async analyzeImport(n){const e=n.name.toLowerCase(),t=e.endsWith(".svg")||e.endsWith(".png");return{supported:t,action:t?"interactive_config":"unsupported",importer:t?"MockImporter":null,features:t?["BITMAP_TRACING"]:[],manifest:t?{title:n.name,natural_size_mm:[100,80],layers:[],warnings:[],errors:[]}:null}},async importFile(n,e={}){const t=ht.find(s=>s.is_active)??ht[0],i={uid:vs("wp"),name:n.name,pos_mm:{x:e.x??100,y:e.y??100},size_mm:{w:40,h:30},angle_deg:0,geometry_revision:1,transform_revision:1};return t.workpieces.push(i),At++,{created:[i.uid],layer_uid:t.uid}},async openProject(n){const t={uid:vs("layer"),name:n.name.replace(/\.[^.]+$/,""),visible:!0,color:"#4d9fff",is_active:!0,wcs:null,rotary_enabled:!1,steps:[],workpieces:[]};return ht.splice(0,ht.length,t),At++,ll()},async saveProject(){return new Blob(["mock project"],{type:"application/octet-stream"})},async exportGcode(){return`; mock gcode
`},async flipItems(n,e){for(const t of Rt)n.includes(t.uid)&&(t.angle_deg=e==="horizontal"?-t.angle_deg:-t.angle_deg+180,t.transform_revision++,t.geometry_revision++);At++,Hn=!0},async alignItems(n,e){const t=Rt.filter(c=>n.includes(c.uid));if(t.length<2)return;const i=e==="left"||e==="right"||e==="center_horizontal",s=c=>i?c.size_mm.w:c.size_mm.h,r=c=>i?c.pos_mm.x:c.pos_mm.y,o=(c,u)=>{i?c.pos_mm.x=u:c.pos_mm.y=u},a=t.map(r),l=t.map(c=>r(c)+s(c));if(e==="left"||e==="top"){const c=Math.min(...a);for(const u of t)o(u,c)}else if(e==="right"||e==="bottom"){const c=Math.max(...l);for(const u of t)o(u,c-s(u))}else{const c=t.map(h=>r(h)+s(h)/2),u=c.reduce((h,d)=>h+d,0)/c.length;for(const h of t)o(h,u-s(h)/2)}for(const c of t)c.transform_revision++,c.geometry_revision++;At++,Hn=!0},async getLayouts(){return[{id:"pixel-perfect",label:"Auto Layout"}]},async runLayout(){let n=20,e=60;for(const t of ht)for(const i of t.workpieces)i.pos_mm={x:n,y:e},i.angle_deg=0,i.transform_revision++,n+=i.size_mm.w+10,n>360&&(n=20,e+=i.size_mm.h+10);At++,Hn=!0,Ns({type:"doc",kind:"history",uids:[],revisions:{tree:At}})},async getMachines(){return gn.map(n=>({...n}))},async getSettings(){return{theme:"dark",language:null,import_dpi:96,cache_budget_bytes:2147483648,watch_source_files:!0,check_for_app_updates:!0,ops_color_mode:"laser",unit_preferences:{length:"mm",speed:"mm/min"}}},async putSettings(n){return{theme:"dark",language:null,import_dpi:96,cache_budget_bytes:2147483648,watch_source_files:!0,check_for_app_updates:!0,ops_color_mode:"laser",unit_preferences:{length:"mm",speed:"mm/min"},...n}},async getMachineSettings(n){if(!gn.some(e=>e.id===n))throw new Error(`mock: unknown machine ${n}`);return{name:"Workbench GRBL",axis_extents:[400,300],origin:"bottom_left",panel_orientation:"native",work_margins:[10,10,10,10],reverse_x_axis:!1,reverse_y_axis:!1,reverse_z_axis:!1,soft_limits_enabled:!0,max_travel_speed:3e3,max_cut_speed:1e3,acceleration:1e3,jog_speed:1e3,dialect_uid:"grbl",heads:[{uid:"mock-head",name:"Laser Head",type:"LaserHead",max_power:1e3,frame_power:0,focus_power:0,frame_speed:0,frame_repeat_count:20,frame_corner_pause:0,spot_size_mm:[.1,.1],focal_distance:0,pointer_offset_enabled:!1,pointer_offset_x:0,pointer_offset_y:0}],rotary_modules:[],rotary_enabled_default:!1,nogo_zones:[{uid:"mock-zone",name:"Clamp",shape:"rect",params:{x:300,y:200,w:80,h:80},enabled:!0}],coordinate_systems:[{name:"G54",label:"WCS 1",offset:[0,0,0]},{name:"G55",label:"",offset:[0,0,0]}]}},async putMachineSettings(n,e){if(!gn.some(t=>t.id===n))throw new Error(`mock: unknown machine ${n}`);return{name:"Workbench GRBL",axis_extents:[400,300],origin:"bottom_left",panel_orientation:"native",work_margins:[10,10,10,10],reverse_x_axis:!1,reverse_y_axis:!1,reverse_z_axis:!1,soft_limits_enabled:!0,max_travel_speed:3e3,max_cut_speed:1e3,acceleration:1e3,jog_speed:1e3,dialect_uid:"grbl",heads:[],rotary_modules:[],rotary_enabled_default:!1,nogo_zones:[],coordinate_systems:[],...e}},async getDrivers(){return[{name:"NoDeviceDriver",maturity:"stable",maturity_warning:""},{name:"GrblSerialDriver",maturity:"stable",maturity_warning:""}]},async getDriverSetup(n){if(!gn.some(e=>e.id===n))throw new Error(`mock: unknown machine ${n}`);return{driver:"GrblSerialDriver",setup:{title:"Setup",fields:[{key:"port",type:"string",kind:"input",label:"Port",description:null,default:"/dev/ttyUSB0",min:null,max:null,step:null,options:null,unit:null,optional:!1},{key:"baudrate",type:"number",kind:"input",label:"Baud Rate",description:null,default:115200,min:9600,max:25e4,step:9600,options:null,unit:null,optional:!1}]},values:{port:"/dev/ttyUSB0",baudrate:115200},precheck_error:null,device_settings:null}},async putDriverSetup(n,e){return await this.getDriverSetup(n),this.getDriverSetup(n).then(t=>({...t,values:{...t.values,...e}}))},async putDriver(n,e){if(!gn.some(t=>t.id===n))throw new Error(`mock: unknown machine ${n}`);if(e!=="NoDeviceDriver"&&e!=="GrblSerialDriver")throw new Error(`mock: unknown driver ${e}`);return this.getDriverSetup(n)},async putHeads(n,e){if(!gn.some(t=>t.id===n))throw new Error(`mock: unknown machine ${n}`);return e.map((t,i)=>({uid:t.uid,name:`Laser Head ${i+1}`,type:"LaserHead",max_power:t.max_power??1e3,frame_power:t.frame_power??0,focus_power:t.focus_power??0,frame_speed:t.frame_speed??0,frame_repeat_count:t.frame_repeat_count??20,frame_corner_pause:t.frame_corner_pause??0,spot_size_mm:t.spot_size_mm??[.1,.1],focal_distance:t.focal_distance??0,pointer_offset_enabled:t.pointer_offset_enabled??!1,pointer_offset_x:t.pointer_offset_x??0,pointer_offset_y:t.pointer_offset_y??0}))},async getMacros(n){if(!gn.some(e=>e.id===n))throw new Error(`mock: unknown machine ${n}`);return[{uid:"mock-macro-1",name:"Home",code:["G28"],enabled:!0}]},async createMacro(n,e){return await this.getMacros(n),{uid:`mock-macro-${Date.now()}`,name:e.name,code:e.code.split(`
`),enabled:e.enabled}},async updateMacro(n,e,t){const s=(await this.getMacros(n)).find(r=>r.uid===e);if(!s)throw new Error("mock: unknown macro");return{...s,...t,code:t.code?t.code.split(`
`):s.code}},async deleteMacro(n,e){if(!(await this.getMacros(n)).some(i=>i.uid===e))throw new Error("mock: unknown macro")},async putWcs(n,e,t,i){return await this.getMachineSettings(n),{name:e,label:i??"",offset:t}},async createNogoZone(n,e){return await this.getMachineSettings(n),{uid:`mock-zone-${Date.now()}`,name:e.name??"No-Go Zone",shape:e.shape,params:e.params,enabled:e.enabled??!0}},async deleteNogoZone(n,e){if(!(await this.getMachineSettings(n)).nogo_zones.some(i=>i.uid===e))throw new Error("mock: unknown zone")},async putRotaryModule(n,e,t){const s=(await this.getMachineSettings(n)).rotary_modules.find(r=>r.uid===e);if(!s)throw new Error("mock: unknown rotary module");return[{...s,...t}]},async getMachineState(){return Ec()},async connectMachine(n){const e=gn.find(t=>t.id===n);if(!e)throw new Error(`mock: unknown machine ${n}`);e.connected=!0,e.is_active&&(qe.connected=!0,dn())},async disconnectMachine(n){const e=gn.find(t=>t.id===n);if(!e)throw new Error(`mock: unknown machine ${n}`);e.connected=!1,e.is_active&&(qe.connected=!1,qe.status="UNKNOWN",dn())},async jog(n,e){en();const t=Wd[n];if(!t)throw new Error(`mock: invalid jog direction ${n}`);const i=qe.machine_pos;if(!i)throw new Error("mock: beam position unknown");Ao(Math.min(400,Math.max(0,(i.x??0)+t[0]*e)),Math.min(300,Math.max(0,(i.y??0)+t[1]*e))),dn()},async homeMachine(){en(),qe.status="IDLE",Ao(0,0),dn()},async moveTo(n,e){en(),Ao(n,e),dn()},async setPower(n){en(),qe.spindle_speed=Math.round(n*100),dn()},async setFocusPower(){en()},async runRaw(){en()},async clearAlarm(){qe.status="IDLE",qe.error=null,dn()},async setWorkOriginHere(){en();const n=qe.machine_pos?.x??0,e=qe.machine_pos?.y??0;qe.wco={x:n,y:e,z:0},qe.work_pos={x:0,y:0,z:0},Li.active_wcs={name:qe.active_wcs??"G54",label:`WCS ${qe.active_wcs??"G54"}`,offset_panel:{x:n,y:e}},dn(),Ns({type:"viewport",data:{...Li}})},async selectWcs(n){en(),qe.active_wcs=n,Li.active_wcs={name:n,label:`WCS ${n}`,offset_panel:{...Li.active_wcs.offset_panel}},Ns({type:"viewport",data:{...Li}})},async syncWcs(){en()},async frame(){en()},async getJobStatus(){return{...mt}},async getOps(){return await ul(60),Zd()},async startJob(){if(en(),mt.active)throw new Error("A job is already running");if(!ht.some(e=>e.workpieces.length>0))throw new Error("Document has no content to run");qe.status="RUN",dn(),Xd()},async holdJob(n){if(en(),!mt.active)throw new Error("No job is running");mt.phase=n?"paused":"running",qe.status=n?"HOLD":"RUN",dn(),to()},async cancelJob(){en(),io(),mt.active=!1,mt.phase="idle",mt.progress_fraction=0,mt.eta_seconds=null,qe.status="IDLE",dn(),to()},getWsUrl(){return"mock://ws"}}}function ul(n){return new Promise(e=>setTimeout(e,n))}function Qd(n){wa.add(n);let e=0;const t=window.setInterval(()=>{e+=.2;const s={x:no.x+20*Math.sin(e/6),y:no.y+15*Math.sin(e/3+.7)};qe.beam_position_panel=s,qe.machine_pos={x:s.x,y:s.y,z:0},qe.work_pos={x:s.x-(qe.wco?.x??0),y:s.y-(qe.wco?.y??0),z:0},n({type:"machine",data:{...Ec()}})},200),i=window.setInterval(()=>{n({type:"notify",level:"info",message:"Mock mode: simulated machine telemetry"})},15e3);return n({type:"job",data:{...mt}}),{close(){wa.delete(n),window.clearInterval(t),window.clearInterval(i),io()}}}function Ye(n){return n&&typeof n=="object"&&!Array.isArray(n)?n:null}function ae(n,e=0){return typeof n=="number"&&Number.isFinite(n)?n:typeof n=="string"&&n!==""&&Number.isFinite(Number(n))?Number(n):e}function Vn(n,e={x:0,y:0}){if(Array.isArray(n)&&n.length>=2)return{x:ae(n[0]),y:ae(n[1])};const t=Ye(n);return t?{x:ae(t.x),y:ae(t.y)}:e}function bn(n){if(n==null||n==="")return null;const e=typeof n=="number"?n:Number(n);return Number.isFinite(e)?e:null}function Co(n){if(Array.isArray(n)&&n.length>=3)return{x:bn(n[0]),y:bn(n[1]),z:bn(n[2])};const e=Ye(n);return e?{x:bn(e.x),y:bn(e.y),z:bn(e.z)}:null}function ef(n){if(Array.isArray(n)&&n.length===6)return[ae(n[0]),ae(n[1]),ae(n[2]),ae(n[3]),ae(n[4]),ae(n[5])];if(Array.isArray(n)&&n.length===2){const e=Array.isArray(n[0])?n[0]:null,t=Array.isArray(n[1])?n[1]:null;if(e&&t)return[ae(e[0]),ae(e[1]),ae(e[2]??0),ae(t[0]),ae(t[1]),ae(t[2]??0)]}return[1,0,0,0,1,0]}function tf(n,e){if(Array.isArray(n)&&n.length>=4){const[i,s,r,o]=n.map(ae);return{x_min:i,y_min:s,x_max:i+r,y_max:s+o}}const t=Ye(n);return t?{x_min:ae(t.x_min),y_min:ae(t.y_min),x_max:ae(t.x_max),y_max:ae(t.y_max)}:e}function nf(n){if(Array.isArray(n)&&n.length>=4)return{x_min:ae(n[0]),y_min:ae(n[1]),x_max:ae(n[2]),y_max:ae(n[3])};const e=Ye(n);return e?{x_min:ae(e.x_min),y_min:ae(e.y_min),x_max:ae(e.x_max),y_max:ae(e.y_max)}:null}function sf(n){if(Array.isArray(n)&&n.length>=4)return{x:ae(n[0]),y:ae(n[1]),w:ae(n[2]),h:ae(n[3])};const e=Ye(n);return e?{x:ae(e.x),y:ae(e.y),w:ae(e.w),h:ae(e.h)}:null}function rf(n){const e=Ye(n.extents);return e?{width_mm:ae(e.width_mm,200),height_mm:ae(e.height_mm,200)}:Array.isArray(n.extents)&&n.extents.length>=2?{width_mm:ae(n.extents[0],200),height_mm:ae(n.extents[1],200)}:{width_mm:200,height_mm:200}}function Sh(n){const e=Ye(n)??{},t=rf(e),i={x_min:0,y_min:0,x_max:t.width_mm,y_max:t.height_mm},s=Ye(e.margins),r=nf(e.soft_limits),o=sf(e.soft_limits_panel),a=Ye(e.active_wcs)??Ye(e.wcs),l=a?{name:typeof a.name=="string"?a.name:"G54",label:typeof a.label=="string"?a.label:typeof a.name=="string"?a.name:"G54",offset_panel:Vn(a.offset_panel)}:null;return{panel_orientation:e.panel_orientation==="rotated_left"||e.panel_orientation==="rotated_right"||e.panel_orientation==="native"?e.panel_orientation:e.orientation==="rotated_left"||e.orientation==="rotated_right"?e.orientation:"native",world_to_panel:ef(e.world_to_panel),x_axis_right:e.x_axis_right!==void 0?!!e.x_axis_right:!0,y_axis_down:!!e.y_axis_down,x_axis_negative:!!e.x_axis_negative,y_axis_negative:!!e.y_axis_negative,extents:t,margins:{left:ae(s?.left),right:ae(s?.right),top:ae(s?.top),bottom:ae(s?.bottom),front:ae(s?.front),back:ae(s?.back)},workarea:tf(e.workarea,i),workarea_origin_panel:Vn(e.workarea_origin_panel??e.workarea_origin),work_origin_panel:Vn(e.work_origin_panel??e.work_origin??e.workarea_origin),active_wcs:l??{name:"G54",label:"G54",offset_panel:{x:0,y:0}},beam_position_panel:e.beam_position_panel!==void 0&&e.beam_position_panel!==null?Vn(e.beam_position_panel):e.beam?Vn(e.beam):null,soft_limits:r,soft_limits_panel:o,nogo_zones:(Array.isArray(e.nogo_zones)?e.nogo_zones:[]).filter(c=>Array.isArray(c)).map(c=>c.map(u=>ae(u))),rotary_active:!!e.rotary_active}}function of(n){return(Array.isArray(n)?n:Array.isArray(Ye(n)?.layouts)?Ye(n).layouts:[]).map(t=>{const i=Ye(t)??{};return{id:String(i.id??""),label:String(i.label??i.id??"")}}).filter(t=>t.id!=="")}function af(n){const e=Ye(n);return e?{code:ae(e.code),title:typeof e.title=="string"?e.title:"Error",description:typeof e.description=="string"?e.description:""}:null}function bh(n){const e=Ye(n)??{},t=e.beam_position_panel!==void 0&&e.beam_position_panel!==null?Vn(e.beam_position_panel):e.beam?Vn(e.beam):null;return{status:typeof e.status=="string"?e.status:"unknown",connected:!!e.connected,machine_pos:Co(e.machine_pos),work_pos:Co(e.work_pos),wco:Co(e.wco),beam_position_panel:t,error:af(e.error),feed_rate:bn(e.feed_rate),spindle_speed:bn(e.spindle_speed),active_wcs:typeof e.active_wcs=="string"?e.active_wcs:null}}function cf(n){return(Array.isArray(n)?n:Array.isArray(Ye(n)?.machines)?Ye(n).machines:[]).map(t=>{const i=Ye(t)??{};return{id:String(i.id??""),name:String(i.name??i.id??""),driver:i.driver===null||i.driver===void 0?"":String(i.driver),connected:!!i.connected,is_active:!!(i.is_active??i.active)}})}function Mh(n){const e=Ye(n)??{},i=Ye(e.job)??e,s=e.active!==void 0?!!e.active:i.phase==="running"||i.phase==="paused",r=typeof i.phase=="string"&&i.phase!==""?i.phase:s?"running":"idle";return{active:s,phase:r,progress_fraction:ae(i.progress_fraction),eta_seconds:i.eta_seconds===null||i.eta_seconds===void 0?null:ae(i.eta_seconds),elapsed_seconds:i.elapsed_seconds===null||i.elapsed_seconds===void 0?null:ae(i.elapsed_seconds)}}function hl(n){const e=Ye(n)??{};return{uid:String(e.uid??""),step_type:String(e.step_type??e.type??"step"),typelabel:typeof e.typelabel=="string"?e.typelabel:void 0,name:String(e.name??"Step"),enabled:e.enabled!==void 0?!!e.enabled:!0,visible:e.visible!==void 0?!!e.visible:void 0,workpiece_uids:Array.isArray(e.workpiece_uids)?e.workpiece_uids.map(String):[],params:Ye(e.params)??{}}}function wh(n){if(Array.isArray(n)&&n.length>=2)return{w:ae(n[0]),h:ae(n[1])};const e=Ye(n);return{w:ae(e?.w),h:ae(e?.h)}}function dl(n){const e=Ye(n)??{};return{uid:String(e.uid??""),name:String(e.name??"Workpiece"),pos_mm:Vn(e.pos_mm),size_mm:wh(e.size_mm),angle_deg:ae(e.angle_deg),geometry_revision:ae(e.geometry_revision,0),transform_revision:ae(e.transform_revision,0)}}function fl(n){const e=Ye(n)??{},t=Array.isArray(e.layers)?e.layers:[],i=typeof e.active_layer_uid=="string"?e.active_layer_uid:null,s=new Map;for(const l of Array.isArray(e.workpieces)?e.workpieces:[]){const c=dl(l);s.set(c.uid,c)}const r=new Map;for(const l of Array.isArray(e.steps)?e.steps:[]){const c=hl(l);r.set(c.uid,c)}const o=t.map(l=>{const c=Ye(l)??{},u=Array.isArray(c.workpieces)?c.workpieces.map(dl):[],h=Array.isArray(c.steps)?c.steps.map(hl):[],d=Array.isArray(c.content_uids)?c.content_uids.map(String):[],f=Array.isArray(c.step_uids)?c.step_uids.map(String):[],g=u.length?u:d.map(m=>s.get(m)).filter(m=>m!==void 0),_=h.length?h:f.map(m=>r.get(m)).filter(m=>m!==void 0);return{uid:String(c.uid??""),name:String(c.name??"Layer"),visible:c.visible!==void 0?!!c.visible:!0,color:typeof c.color=="string"?c.color:"#4d9fff",is_active:!!c.is_active||c.uid===i,wcs:typeof c.wcs=="string"?c.wcs:null,rotary_enabled:c.rotary_enabled!==void 0?!!c.rotary_enabled:void 0,steps:_,workpieces:g}}),a=(Array.isArray(e.stock_items)?e.stock_items:[]).map(l=>{const c=Ye(l)??{};return{uid:String(c.uid??""),name:String(c.name??"Stock"),visible:c.visible!==void 0?!!c.visible:!0,color:typeof c.color=="string"||c.color===null?c.color:null,pos_mm:Vn(c.pos_mm),size_mm:wh(c.size_mm),angle_deg:ae(c.angle_deg),geometry_revision:ae(c.geometry_revision,0),transform_revision:ae(c.transform_revision,0)}});return{rev:ae(e.rev,-1),assets_count:ae(e.assets_count,s.size),layers:o,stock_items:a}}function lf(n){const e=Ye(n)??{},t=Ye(e.outline);return{rev:ae(e.rev,0),polylines:(Array.isArray(e.polylines)?e.polylines:[]).filter(i=>Array.isArray(i)).map(i=>i.map(s=>ae(s))),outline:{x_min:ae(t?.x_min),y_min:ae(t?.y_min),x_max:ae(t?.x_max),y_max:ae(t?.y_max)}}}const uf=["input","switch","slider","length","speed","angle","textarea","choice"];function hf(n){const e=Ye(n)??{},t=e.type==="boolean"||e.type==="number"?e.type:"string",i=uf.includes(e.kind)?e.kind:"input",s=Array.isArray(e.options)?e.options.map(o=>{const a=Ye(o)??{};return{value:String(a.value??""),label:String(a.label??a.value??"")}}).filter(o=>o.value!==""):null;return{key:String(e.key??""),type:t,kind:i,label:String(e.label??e.key??""),description:typeof e.description=="string"?e.description:null,default:e.default,min:e.min===null||e.min===void 0?null:ae(e.min),max:e.max===null||e.max===void 0?null:ae(e.max),step:e.step===null||e.step===void 0?null:ae(e.step),options:s,unit:typeof e.unit=="string"?e.unit:null,optional:!!e.optional}}function df(n){const e=Ye(n)??{},t=(Array.isArray(e.groups)?e.groups:[]).map(i=>{const s=Ye(i)??{};return{title:String(s.title??""),fields:(Array.isArray(s.fields)?s.fields:[]).map(hf)}});return{step_type:String(e.step_type??""),groups:t}}function ff(n){const e=Ye(n)??{};return{step_type:String(e.step_type??""),typelabel:String(e.typelabel??e.step_type??""),icon:typeof e.icon=="string"?e.icon:null,required_machine_caps:Array.isArray(e.required_machine_caps)?e.required_machine_caps.map(String):[]}}const pf=["direct_load","interactive_config","unsupported"];function mf(n){const e=Ye(n)??{},t=pf.includes(e.action)?e.action:"unsupported",i=Ye(e.manifest);return{supported:!!e.supported,action:t,importer:typeof e.importer=="string"?e.importer:null,features:Array.isArray(e.features)?e.features.map(String):[],manifest:i?{title:String(i.title??""),natural_size_mm:Array.isArray(i.natural_size_mm)?[ae(i.natural_size_mm[0]),ae(i.natural_size_mm[1])]:null,layers:(Array.isArray(i.layers)?i.layers:[]).map(s=>{const r=Ye(s)??{};return{id:String(r.id??""),name:String(r.name??""),default_active:!!r.default_active,feature_count:ae(r.feature_count,0)}}),warnings:Array.isArray(i.warnings)?i.warnings.map(String):[],errors:Array.isArray(i.errors)?i.errors.map(String):[]}:null}}function gf(n){if(!Array.isArray(n))return[];const e=n[0];if(typeof e=="number"||typeof e=="string")return n.map(i=>ae(i));const t=[];for(const i of n){if(Array.isArray(i)&&i.length>=2){t.push(ae(i[0]),ae(i[1]));continue}const s=Ye(i);s&&t.push(ae(s.x),ae(s.y))}return t}function _f(n){const e=Ye(n)??{};return{segments:(Array.isArray(e.segments)?e.segments:[]).map(i=>{const s=Ye(i)??{};return{points:gf(s.points),power:ae(s.power),rate:bn(s.rate),cutting:!!s.cutting}}).filter(i=>i.points.length>=4||!i.cutting&&i.points.length>=2),estimated_seconds:bn(e.estimated_seconds),distance_mm:ae(e.distance_mm),generation_id:String(e.generation_id??"")}}class Ki extends Error{constructor(e,t){super(t),this.status=e,this.name="ApiError"}status}const pl="/api/v1";class vf{token;constructor(){this.token=typeof window>"u"?null:new URLSearchParams(window.location.search).get("token")}getSystemInfo(){return this.get("/system/info")}getViewport(){return this.get("/viewport").then(Sh)}getDocTree(){return this.get("/doc/tree").then(fl)}getItemGeometry(e,t=.05){const i=encodeURIComponent(t);return this.get(`/items/${e}/geometry?tolerance=${i}`).then(lf)}async getItemTexture(e,t){const i=await fetch(this.url(`/items/${e}/texture.png?rev=${t}`),{headers:this.headers()});if(!i.ok)return null;const s=await i.blob();return URL.createObjectURL(s)}getDocHistory(){return this.get("/doc/history")}postUndo(){return this.post("/doc/undo")}postRedo(){return this.post("/doc/redo")}postNewDoc(){return this.post("/doc/new")}putItemTransform(e,t){return this.put(`/items/${e}/transform`,{pos_mm:[t.pos_mm.x,t.pos_mm.y],size_mm:[t.size_mm.w,t.size_mm.h],angle_deg:t.angle_deg})}flipItems(e,t){return this.postJson("/items/flip",{uids:e,axis:t}).then(()=>{})}alignItems(e,t){return this.postJson("/items/align",{uids:e,mode:t}).then(()=>{})}getLayouts(){return this.get("/layouts").then(of)}runLayout(e,t){return this.postJson("/layout",{strategy:e,...t?{params:t}:{}}).then(()=>{})}getSettings(){return this.get("/settings")}putSettings(e){return this.putJson("/settings",{values:e})}getMachineSettings(e){return this.get(`/machines/${encodeURIComponent(e)}/settings`)}putMachineSettings(e,t){return this.putJson(`/machines/${encodeURIComponent(e)}/settings`,{values:t})}getDrivers(){return this.get("/drivers")}getDriverSetup(e){return this.get(`/machines/${encodeURIComponent(e)}/driver-setup`)}putDriverSetup(e,t){return this.putJson(`/machines/${encodeURIComponent(e)}/driver-setup`,{values:t})}putDriver(e,t){return this.putJson(`/machines/${encodeURIComponent(e)}/driver`,{driver:t})}putHeads(e,t){return this.putJson(`/machines/${encodeURIComponent(e)}/heads`,{heads:t})}getMacros(e){return this.get(`/machines/${encodeURIComponent(e)}/macros`).then(t=>{const i=t;return Array.isArray(i?.macros)?i.macros:[]})}createMacro(e,t){return this.postJson(`/machines/${encodeURIComponent(e)}/macros`,t)}updateMacro(e,t,i){return this.putJson(`/machines/${encodeURIComponent(e)}/macros/${encodeURIComponent(t)}`,i)}deleteMacro(e,t){return this.requestVoid("DELETE",`/machines/${encodeURIComponent(e)}/macros/${encodeURIComponent(t)}`)}putWcs(e,t,i,s){return this.putJson(`/machines/${encodeURIComponent(e)}/wcs/${encodeURIComponent(t)}`,{offset:i,...s!==void 0?{label:s}:{}})}createNogoZone(e,t){return this.postJson(`/machines/${encodeURIComponent(e)}/nogo-zones`,t)}deleteNogoZone(e,t){return this.requestVoid("DELETE",`/machines/${encodeURIComponent(e)}/nogo-zones/${encodeURIComponent(t)}`)}putRotaryModule(e,t,i){return this.putJson(`/machines/${encodeURIComponent(e)}/rotary-modules/${encodeURIComponent(t)}`,i)}putLayer(e,t){return this.put(`/layers/${e}`,t)}postLayer(){return this.postJson("/layers")}deleteLayer(e){return this.requestVoid("DELETE",`/layers/${e}`)}activateLayer(e){return this.post(`/layers/${e}/activate`)}reorderLayers(e){return this.postJson("/layers/reorder",{uids:e}).then(()=>{})}getStepTypes(){return this.get("/steps").then(e=>(Array.isArray(e.steps)?e.steps:[]).map(ff))}getStepSchema(e){return this.get(`/steps/${e}/schema`).then(df)}postStep(e,t){return this.postJson(`/layers/${e}/steps`,t)}putStep(e,t){return this.putJson(`/steps/${e}`,t)}deleteStep(e){return this.requestVoid("DELETE",`/steps/${e}`)}deleteItems(e){return this.postJson("/items/delete",{uids:e}).then(()=>{})}duplicateItems(e){return this.postJson("/items/duplicate",{uids:e})}analyzeImport(e){return this.postForm("/import/analyze",e).then(mf)}importFile(e,t={}){const i=new URLSearchParams;t.spec&&i.set("spec",JSON.stringify(t.spec)),t.x!==void 0&&t.y!==void 0&&(i.set("x",String(t.x)),i.set("y",String(t.y)));const s=i.toString();return this.postForm(`/import${s?`?${s}`:""}`,e)}openProject(e){return this.postForm("/project/open",e).then(fl)}async saveProject(){const e=await fetch(this.url("/project/save"),{headers:this.headers()});if(!e.ok)throw new Ki(e.status,`GET /project/save failed (${e.status})`);return e.blob()}async exportGcode(){const e=await fetch(this.url("/project/gcode"),{headers:this.headers()});if(!e.ok)throw new Ki(e.status,`GET /project/gcode failed (${e.status})`);return e.text()}getMachines(){return this.get("/machines").then(cf)}getMachineState(){return this.get("/machine/state").then(bh)}connectMachine(e){return this.post(`/machines/${encodeURIComponent(e)}/connect`)}disconnectMachine(e){return this.post(`/machines/${encodeURIComponent(e)}/disconnect`)}jog(e,t,i){return this.postJson("/machine/jog",{direction:e,distance_mm:t,...i!==void 0&&i>0?{speed:Math.round(i)}:{}}).then(()=>{})}homeMachine(e){return this.postJson("/machine/home",{axes:e??[]}).then(()=>{})}moveTo(e,t,i,s){return this.postJson("/machine/move-to",{x:e,y:t,...i!==void 0?{z:i}:{},...s!==void 0&&s>0?{speed:s}:{}}).then(()=>{})}setPower(e){return this.postJson("/machine/set-power",{percent:e}).then(()=>{})}setFocusPower(e){return this.postJson("/machine/set-focus-power",{percent:e}).then(()=>{})}runRaw(e){return this.postJson("/machine/run-raw",{gcode:e}).then(()=>{})}clearAlarm(){return this.post("/machine/clear-alarm")}setWorkOriginHere(e){return this.postJson("/machine/set-work-origin-here",{axes:e??[]}).then(()=>{})}selectWcs(e){return this.postJson("/machine/select-wcs",{slot:e}).then(()=>{})}syncWcs(){return this.post("/machine/sync-wcs")}frame(){return this.postJson("/machine/frame",{include_overscan:!1}).then(()=>{})}getJobStatus(){return this.get("/job/status").then(Mh)}getOps(e=.05){const t=encodeURIComponent(e);return this.get(`/ops?tolerance=${t}`).then(_f)}startJob(){return this.post("/job/start")}holdJob(e){return this.postJson("/job/hold",{hold:e}).then(()=>{})}cancelJob(){return this.post("/job/cancel")}getWsUrl(){const e=typeof window<"u"&&window.location.protocol==="https:"?"wss:":"ws:",t=typeof window>"u"?"":window.location.host,i=this.token?`?token=${encodeURIComponent(this.token)}`:"";return`${e}//${t}${pl}/ws${i}`}url(e){return`${pl}${e}`}headers(){const e={};return this.token&&(e.Authorization=`Bearer ${this.token}`),e}async get(e){const t=await fetch(this.url(e),{headers:this.headers()});return this.parse(t,"GET",e)}async post(e){const t=await fetch(this.url(e),{method:"POST",headers:this.headers()});await this.parse(t,"POST",e)}async postJson(e,t){const i=await fetch(this.url(e),{method:"POST",headers:{"Content-Type":"application/json",...this.headers()},body:JSON.stringify(t??{})});return this.parse(i,"POST",e)}async put(e,t){await this.putJson(e,t)}async putJson(e,t){const i=await fetch(this.url(e),{method:"PUT",headers:{"Content-Type":"application/json",...this.headers()},body:JSON.stringify(t)});return this.parse(i,"PUT",e)}async requestVoid(e,t){const i=await fetch(this.url(t),{method:e,headers:this.headers()});await this.parse(i,e,t)}async postForm(e,t){const i=new FormData;i.append("file",t,t.name);const s=await fetch(this.url(e),{method:"POST",headers:this.headers(),body:i});return this.parse(s,"POST",e)}async parse(e,t,i){if(!e.ok)throw await this.errorFrom(e,t,i);const s=await e.text();if(s)return JSON.parse(s)}async errorFrom(e,t,i){let s="";try{const o=await e.json();typeof o.detail=="string"?s=o.detail:Array.isArray(o.detail)&&(s=o.detail.map(a=>typeof a=="string"?a:typeof a?.msg=="string"?a.msg:JSON.stringify(a)).join("; "))}catch{return new Ki(e.status,`${t} ${i} failed (${e.status})`)}const r=s?`${s} (${e.status})`:`${t} ${i} failed (${e.status})`;return new Ki(e.status,r)}}function xf(){return _h()?Kd():new vf}const le=xf(),yf=n=>{let e;const t=new Set,i=(c,u)=>{const h=typeof c=="function"?c(e):c;if(!Object.is(h,e)){const d=e;e=u??(typeof h!="object"||h===null)?h:Object.assign({},e,h),t.forEach(f=>f(e,d))}},s=()=>e,a={setState:i,getState:s,getInitialState:()=>l,subscribe:c=>(t.add(c),()=>t.delete(c))},l=e=n(i,s,a);return a},fs=(n=>yf),We=fs()(()=>({loaded:!1,rev:-1,layers:[],stocks:[],assetsCount:0,selection:[],expanded:[],canUndo:!1,canRedo:!1,dirty:!1,undoLabels:[],redoLabels:[],rolling:!1}));function Sf(n,e){const t=n.revisions?n.revisions.tree:void 0;return{refetchTree:t===void 0||t!==e,refetchHistory:!0,changedUids:n.uids??[]}}function bf(n,e){return n===void 0||n!==e.geometry_revision}function Mf(n,e){return n.find(t=>t.uid===e)}function Tc(n,e){for(const t of n){const i=t.workpieces.find(s=>s.uid===e);if(i)return{layer:t,workpiece:i}}}function wf(n,e){for(const t of n){const i=t.steps.find(s=>s.uid===e);if(i)return{layer:t,step:i}}}function Ef(n,e){const t=[];for(const i of e){const s=Tc(n,i);s&&t.push(s.workpiece)}return t}function Tf(n,e){const t=new Set;for(const i of n){t.add(i.uid);for(const s of i.workpieces)t.add(s.uid);for(const s of i.steps)t.add(s.uid)}return e.filter(i=>t.has(i))}function ml(n){We.setState(e=>({loaded:!0,rev:n.rev,layers:n.layers,stocks:n.stock_items??[],assetsCount:n.assets_count,selection:Tf(n.layers,e.selection)}))}function Af(n){We.setState({canUndo:n.can_undo,canRedo:n.can_redo,undoLabels:n.undo_labels,redoLabels:n.redo_labels,dirty:n.dirty})}function gl(n){We.setState({rolling:n})}function ei(n,e=!1){We.setState(t=>{if(!e)return{selection:[...n]};const i=new Set(t.selection);for(const s of n)i.has(s)?i.delete(s):i.add(s);return{selection:[...i]}})}function Cf(){We.setState({selection:[]})}function Rf(n){We.setState(e=>({expanded:e.expanded.includes(n)?e.expanded.filter(t=>t!==n):[...e.expanded,n]}))}function Pf(n,e){return n.expanded.includes(e)}function En(n,e){We.setState(t=>({layers:t.layers.map(i=>i.uid===n?{...i,...e}:i)}))}function Lf(n){We.setState(e=>({layers:e.layers.map(t=>({...t,is_active:t.uid===n}))}))}function Df(n){We.setState(e=>{const t=new Map(e.layers.map(s=>[s.uid,s])),i=[];for(const s of n){const r=t.get(s);r&&i.push(r)}for(const s of e.layers)i.includes(s)||i.push(s);return{layers:i}})}function If(n){return n instanceof HTMLElement?n instanceof HTMLInputElement||n instanceof HTMLTextAreaElement||n instanceof HTMLSelectElement||n.isContentEditable:!1}class Nf{constructor(e){this.deps=e}deps;selection(){return We.getState().selection.filter(e=>Tc(We.getState().layers,e))}run(e,t){e().then(()=>this.deps.refreshDoc()).catch(i=>{this.deps.onError(`${t} failed: ${Uf(i)}`)})}deleteSelected(){const e=this.selection();e.length!==0&&this.run(()=>this.deps.api.deleteItems(e),"Delete")}duplicateSelected(){const e=this.selection();e.length!==0&&this.run(async()=>{const t=await this.deps.api.duplicateItems(e);return t.created.length>0&&ei(t.created),t},"Duplicate")}flipSelected(e){const t=this.selection();t.length!==0&&this.run(()=>this.deps.api.flipItems(t,e),"Flip")}alignSelected(e){const t=this.selection();t.length!==0&&this.run(()=>this.deps.api.alignItems(t,e),"Align")}undo(){We.getState().canUndo&&this.run(()=>this.deps.api.postUndo(),"Undo")}redo(){We.getState().canRedo&&this.run(()=>this.deps.api.postRedo(),"Redo")}handleKeydown(e){if(!If(e.target)){if(e.ctrlKey||e.metaKey){const t=e.key.toLowerCase();t==="z"&&e.shiftKey?(e.preventDefault(),this.redo()):t==="z"?(e.preventDefault(),this.undo()):t==="y"?(e.preventDefault(),this.redo()):t==="d"&&(e.preventDefault(),this.duplicateSelected());return}(e.key==="Delete"||e.key==="Backspace")&&(e.preventDefault(),this.deleteSelected())}}}function Uf(n){return n instanceof Error?n.message:String(n)}const Ac=fs()(()=>({transformPreview:null}));function Of(n){Ac.setState({transformPreview:n})}function Ff(){Ac.setState({transformPreview:null})}const Et=fs()(()=>({machines:[],viewport:null,machineState:null,activeMachineId:null}));function _l(n){Et.setState({viewport:n})}function kf(n){Et.setState(e=>({machines:n,activeMachineId:e.activeMachineId??n.find(t=>t.is_active)?.id??n[0]?.id??null}))}function Ea(n){Et.setState({machineState:n})}const Yt=fs()(()=>({active:!1,phase:"idle",progressFraction:0,etaSeconds:null,elapsedSeconds:null}));function vl(n){Yt.setState({active:n.active,phase:n.phase,progressFraction:n.progress_fraction,etaSeconds:n.eta_seconds,elapsedSeconds:n.elapsed_seconds})}const ss=1e-6,zf=25;function Bf(n){const e=[];let t=0;for(const i of n){const s=[0];let r=0;for(let a=2;a+1<i.points.length;a+=2)r+=Math.hypot(i.points[a]-i.points[a-2],i.points[a+1]-i.points[a-1]),s.push(r);const o=t;t+=r,e.push({...i,startDistance:o,endDistance:t,cumulative:s})}return{segments:e,totalDistance:t}}function mo(n,e){return Math.min(Math.max(e,0),n.totalDistance)}function Hf(n,e){if(n.segments.length===0)return null;const t=mo(n,e);let i=0,s=n.segments.length-1;for(;i<s;){const r=i+s>>1;n.segments[r].endDistance>=t-ss?s=r:i=r+1}return n.segments[i]}function Gf(n,e){const t=n.points;if(t.length<4)return{x:t[0]??0,y:t[1]??0};const i=Math.min(Math.max(e-n.startDistance,0),n.endDistance-n.startDistance),s=n.cumulative;let r=0,o=s.length-2;for(;r<o;){const f=r+o>>1;s[f+1]>=i?o=f:r=f+1}const a=s[r+1]-s[r],l=a>ss?(i-s[r])/a:0,c=t[r*2],u=t[r*2+1],h=t[r*2+2],d=t[r*2+3];return{x:c+(h-c)*l,y:u+(d-u)*l}}function Vf(n,e){const t=Hf(n,e);if(!t)return null;const i=Gf(t,mo(n,e));return{x:i.x,y:i.y,segmentIndex:n.segments.indexOf(t),power:t.power,cutting:t.cutting}}function Wf(n,e){let t=n.totalDistance;for(const i of n.segments)i.endDistance-i.startDistance<=ss||i.endDistance>e+ss&&i.endDistance<t&&(t=i.endDistance);return t}function Xf(n,e){let t=0;for(const i of n.segments)i.endDistance-i.startDistance<=ss||i.startDistance<e-ss&&i.startDistance>t&&(t=i.startDistance);return t}function $f(n,e,t){return mo(n,e+t)}function Yf(n,e,t,i){return(i!==null&&i>0?n.totalDistance/i:zf)*Math.max(e,0)*Math.max(t,0)}function qf(n){let e=!0,t=performance.now();const i=s=>{if(!e)return;const r=Math.min(Math.max((s-t)/1e3,0),.25);t=s,n(r),window.requestAnimationFrame(i)};return window.requestAnimationFrame(i),()=>{e=!1}}const jf=.05,Zf=[1,10,60],lt=fs()(()=>({available:!1,fetching:!1,generationId:null,segments:[],estimatedSeconds:null,distanceMm:null,track:null,playing:!1,scrubbing:!1,positionMm:0,speed:1}));function Jf(n,e){return n!==e}function Kf(n){const e=lt.getState();e.available&&e.generationId===n.generation_id||lt.setState({available:!0,generationId:n.generation_id,segments:n.segments,estimatedSeconds:n.estimated_seconds,distanceMm:n.distance_mm,track:Bf(n.segments),playing:!1,scrubbing:!1,positionMm:0})}function Qf(){lt.setState({available:!1,generationId:null,segments:[],estimatedSeconds:null,distanceMm:null,track:null,playing:!1,scrubbing:!1,positionMm:0})}async function Ro(n=le){lt.setState({fetching:!0});try{Kf(await n.getOps(jf))}catch(e){e instanceof Ki&&e.status===404&&Qf()}finally{lt.setState({fetching:!1})}}function Cc(){return lt.getState().track!==null&&!Yt.getState().active}function ep(){if(!Cc())return;const n=lt.getState(),e=n.track!==null&&n.positionMm>=n.track.totalDistance;lt.setState({playing:!0,positionMm:e?0:n.positionMm})}function tp(){lt.setState({playing:!1})}function np(n){const e=lt.getState().track;e&&lt.setState({positionMm:mo(e,n)})}function ip(n){!Number.isFinite(n)||n<=0||lt.setState({speed:n})}function rr(n){lt.getState().scrubbing!==n&&lt.setState({scrubbing:n})}function sp(){const n=lt.getState();Cc()&&lt.setState({positionMm:Wf(n.track,n.positionMm)})}function rp(){const n=lt.getState();Cc()&&lt.setState({positionMm:Xf(n.track,n.positionMm)})}function op(){return qf(n=>{const e=lt.getState();if(!e.playing||!e.track||Yt.getState().active)return;const t=Yf(e.track,e.speed,n,e.estimatedSeconds),i=$f(e.track,e.positionMm,t);i>=e.track.totalDistance?lt.setState({positionMm:i,playing:!1}):lt.setState({positionMm:i})})}function Pt(n,e=0){return{x:n.x,y:e,z:-n.y}}function ap(n,e,t){return{x:n,y:-t}}function cp(n,e){return e?n.every((t,i)=>t===e[i]):!1}function ii(n,e){return{x:n[0]*e.x+n[1]*e.y+n[2],y:n[3]*e.x+n[4]*e.y+n[5]}}function lp(n){const[e,t,i,s,r,o]=n,a=e*r-t*s;return a===0?[1,0,0,0,1,0]:[r/a,-t/a,-(r/a*i+-t/a*o),-s/a,e/a,-(-s/a*i+e/a*o)].map(c=>c===0?0:c)}function xl(n,e){return ii(lp(n),e)}function Ta(n,e){const t=new Array(e.length);for(let i=0;i+1<e.length;i+=2){const s=ii(n,{x:e[i],y:e[i+1]});t[i]=s.x,t[i+1]=s.y}return t}function up(n,e){let t=!1;const i=Math.floor(e.length/2);for(let s=0,r=i-1;s<i;r=s++){const o=e[s*2],a=e[s*2+1],l=e[r*2],c=e[r*2+1];a>n.y!=c>n.y&&n.x<(l-o)*(n.y-a)/(c-a)+o&&(t=!t)}return t}const Aa="rayforge.webui.view";function or(n,e){if(typeof localStorage>"u")return e;try{const t=localStorage.getItem(Aa);if(!t)return e;const i=JSON.parse(t);return typeof i[n]=="boolean"?i[n]:e}catch{return e}}function go(n,e){if(!(typeof localStorage>"u"))try{const t=localStorage.getItem(Aa),i=t?JSON.parse(t):{};localStorage.setItem(Aa,JSON.stringify({...i,[n]:e}))}catch{return}}const je=fs()(()=>({activeLeftTab:"document",toasts:[],systemInfo:null,wsStatus:"closed",mockMode:_h(),projectName:"Untitled",showToolpaths:!0,showTravelMoves:or("showTravelMoves",!0),showWorkpieceImages:or("showWorkpieceImages",!0),showNogoZones:or("showNogoZones",!0),showStock:or("showStock",!0)}));let hp=0;function _e(n,e){const t=++hp;return je.setState(i=>({toasts:[...i.toasts,{id:t,level:n,message:e}].slice(-5)})),t}function Eh(n){je.setState(e=>({toasts:e.toasts.filter(t=>t.id!==n)}))}function Th(n){je.setState({activeLeftTab:n})}function dp(n){je.setState({systemInfo:n})}function yl(n){je.setState({wsStatus:n})}function Sl(n){je.setState({projectName:n})}function Ah(n){je.setState({showToolpaths:n})}function Ch(n){go("showTravelMoves",n),je.setState({showTravelMoves:n})}function Rh(n){go("showWorkpieceImages",n),je.setState({showWorkpieceImages:n})}function Ph(n){go("showNogoZones",n),je.setState({showNogoZones:n})}function fp(n){go("showStock",n),je.setState({showStock:n})}const pp=1e4,mp=500;function gp(n,e,t){let i=null,s=0,r=!1,o=!1,a;function l(){if(r)return;const u=Math.min(pp,mp*2**Math.min(s,4));s++,a=setTimeout(c,u)}function c(){if(!r){e.onStatus("connecting");try{i=new WebSocket(n)}catch{l();return}i.onopen=()=>{const u=o;o=!0,s=0,e.onStatus("open"),u&&t?.()},i.onmessage=u=>{try{e.onEvent(JSON.parse(String(u.data)))}catch{return}},i.onerror=()=>{i?.close()},i.onclose=()=>{e.onStatus("closed"),l()}}}return c(),{close(){r=!0,a!==void 0&&clearTimeout(a),i&&(i.onclose=null,i.close()),e.onStatus("closed")}}}function _p(n,e){switch(n.type){case"doc":e.onDoc?.(n);break;case"viewport":e.onViewport?.(n.data);break;case"machine":e.onMachine?.(n.data);break;case"job":e.onJob?.(n.data);break;case"notify":e.onNotify?.(n.level,n.message);break}}function bl(n,e){const t=URL.createObjectURL(n),i=document.createElement("a");i.href=t,i.download=e,document.body.appendChild(i),i.click(),i.remove(),window.setTimeout(()=>URL.revokeObjectURL(t),2e3)}function Lh(n){const{active:e,phase:t,connected:i,hasContent:s}=n;if(!e)return{kind:"idle",phaseLabel:"Idle",startEnabled:i&&s,holdEnabled:!1,holdLabel:"Pause",holdIcon:"pause",cancelEnabled:!1,frameEnabled:i};const r=t==="paused";return{kind:r?"paused":"running",phaseLabel:r?"Paused":"Running",startEnabled:!1,holdEnabled:!0,holdLabel:r?"Resume":"Pause",holdIcon:r?"play":"pause",cancelEnabled:!0,frameEnabled:!1}}function vp(n){if(n===null||!Number.isFinite(n)||n<0)return"--:--";const e=Math.round(n),t=Math.floor(e/3600),i=Math.floor(e%3600/60),s=e%60,r=String(i).padStart(2,"0"),o=String(s).padStart(2,"0");return t>0?`${t}:${r}:${o}`:`${r}:${o}`}function E(n,e,t=[]){const i=document.createElement(n);e&&(i.className=e);for(const s of t)i.append(s);return i}function Cn(n){for(;n.firstChild;)n.removeChild(n.firstChild)}function Ml(n,e=1){return n==null||Number.isNaN(n)?"–":n.toFixed(e)}function Ca(n){if(n==null||Number.isNaN(n))return"–";if(n<60)return`${n.toFixed(1)}s`;const e=Math.floor(n/60),t=Math.round(n-e*60),i=t>=60?59:t;return`${e}m ${String(i).padStart(2,"0")}s`}const xp={pointer:'<path d="M4 3l7.5 18 2.4-7.4L21 11z"/>',move:'<path d="M12 2v20M2 12h20"/><path d="M9 5l3-3 3 3M9 19l3 3 3-3M5 9l-3 3 3 3M19 9l3 3-3 3"/>',rotate:'<path d="M21 12a9 9 0 1 1-2.64-6.36L21 8"/><path d="M21 3v5h-5"/>',scale:'<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>',upload:'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5M12 3v12"/>',frame:'<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>',fit:'<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M16 21h3a2 2 0 0 0 2-2v-3M8 21H5a2 2 0 0 1-2-2v-3"/><circle cx="12" cy="12" r="1.6"/>',viewTop:'<rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="12" cy="12" r="1.6"/><path d="M12 4v5M12 15v5M4 12h5M15 12h5"/>',viewFront:'<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 15h16"/>',viewIso:'<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.3 7l8.7 5 8.7-5M12 22V12"/>',play:'<path d="M6 4l14 8-14 8z"/>',pause:'<path d="M7 4h3v16H7zM14 4h3v16h-3z"/>',stop:'<rect x="5" y="5" width="14" height="14" rx="2"/>',stepForward:'<path d="M5 4l10 8-10 8z"/><path d="M19 5v14"/>',stepBack:'<path d="M19 4L9 12l10 8z"/><path d="M5 5v14"/>',undo:'<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-15-6.7L3 13"/>',redo:'<path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 15-6.7L21 13"/>',filePlus:'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M12 18v-6M9 15h6"/>',save:'<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',folder:'<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',eye:'<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',eyeOff:'<path d="M17.94 17.94A10.5 10.5 0 0 1 12 20c-7 0-11-8-11-8a19.8 19.8 0 0 1 5.06-5.94M9.9 4.24A9.7 9.7 0 0 1 12 4c7 0 11 8 11 8a19.8 19.8 0 0 1-3.22 4.31"/><path d="M1 1l22 22"/>',chevronRight:'<path d="M9 18l6-6-6-6"/>',chevronDown:'<path d="M6 9l6 6 6-6"/>',chevronUp:'<path d="M18 15l-6-6-6 6"/>',chevronLeft:'<path d="M15 18l-6-6 6-6"/>',home:'<path d="M3 10.5L12 3l9 7.5"/><path d="M5 9.5V21h5v-6h4v6h5V9.5"/>',crosshairSmall:'<circle cx="12" cy="12" r="5"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>',terminal:'<path d="M4 17l6-5-6-5"/><path d="M12 19h8"/>',kebab:'<circle cx="12" cy="5" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="12" cy="19" r="1.4"/>',layers:'<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/>',box:'<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.3 7l8.7 5 8.7-5M12 22V12"/>',zap:'<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>',machine:'<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>',link:'<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',sliders:'<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3"/><path d="M1 14h6M9 8h6M17 16h6"/>',crosshair:'<circle cx="12" cy="12" r="9"/><path d="M23 12h-4M5 12H1M12 1v4M12 19v4"/>',grid:'<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',x:'<path d="M18 6L6 18M6 6l12 12"/>',check:'<path d="M20 6L9 17l-5-5"/>',trash:'<path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M10 11v6M14 11v6"/>',copy:'<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',alert:'<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',info:'<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',plug:'<path d="M12 22v-5M9 8V2M15 8V2M18 8v3a6 6 0 0 1-12 0V8z"/>',search:'<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',zoomIn:'<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6M11 8v6"/>',zoomOut:'<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/>',flipH:'<path d="M12 3v18"/><path d="M8 8L4 12l4 4"/><path d="M16 8l4 4-4 4"/>',flipV:'<path d="M3 12h18"/><path d="M8 8l4-4 4 4"/><path d="M8 16l4 4 4-4"/>',alignLeft:'<path d="M4 3v18"/><rect x="8" y="5" width="12" height="5" rx="1"/><rect x="8" y="14" width="8" height="5" rx="1"/>',alignCenterH:'<path d="M12 3v18"/><rect x="4" y="5" width="16" height="5" rx="1"/><rect x="7" y="14" width="10" height="5" rx="1"/>',alignRight:'<path d="M20 3v18"/><rect x="4" y="5" width="12" height="5" rx="1"/><rect x="8" y="14" width="8" height="5" rx="1"/>',alignTop:'<path d="M3 4h18"/><rect x="5" y="8" width="5" height="12" rx="1"/><rect x="14" y="8" width="5" height="8" rx="1"/>',alignCenterV:'<path d="M3 12h18"/><rect x="5" y="4" width="5" height="16" rx="1"/><rect x="14" y="7" width="5" height="10" rx="1"/>',alignBottom:'<path d="M3 20h18"/><rect x="5" y="4" width="5" height="12" rx="1"/><rect x="14" y="8" width="5" height="8" rx="1"/>',plugOff:'<path d="M12 22v-5M9 8V2M15 8V2M18 8v3c0 1-.2 1.9-.6 2.7"/><path d="M6.6 6.6A6 6 0 0 0 6 11v3"/><path d="M4 4l16 16"/>',refresh:'<path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.5 9a9 9 0 0 1 14.9-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>'};function Gt(n,e=16){const t=document.createElementNS("http://www.w3.org/2000/svg","svg");return t.setAttribute("viewBox","0 0 24 24"),t.setAttribute("width",String(e)),t.setAttribute("height",String(e)),t.setAttribute("fill","none"),t.setAttribute("stroke","currentColor"),t.setAttribute("stroke-width","1.8"),t.setAttribute("stroke-linecap","round"),t.setAttribute("stroke-linejoin","round"),t.classList.add("icon",`icon-${n}`),t.innerHTML=xp[n]??"",t}function yp(n){const e=E("div","floating-toolbar ribbon");let t="select",i="model";function s(l){const c=E("button","ribbon-btn");c.type="button",c.title=l.title,c.appendChild(Gt(l.iconName,18));const u=E("span","ribbon-btn-label");return u.textContent=l.label,c.appendChild(u),l.active&&c.classList.add("active"),l.disabled&&(c.disabled=!0),l.onClick&&c.addEventListener("click",()=>l.onClick?.(c)),c}function r(l){const c=E("div","ribbon-tabs");for(const u of l){const h=E("button","ribbon-tab");h.type="button",h.textContent=u.label,u.id===i&&h.classList.add("active"),h.addEventListener("click",()=>{i=u.id,a()}),c.appendChild(h)}return c}function o(l){const c=E("div","ribbon-body");return l.groups().forEach((u,h)=>{h>0&&c.appendChild(E("div","ribbon-sep"));const d=E("div","ribbon-group"),f=E("div","ribbon-items");for(const _ of u.items)f.appendChild(s(_));const g=E("div","ribbon-group-label");g.textContent=u.label,d.append(f,g),c.appendChild(d)}),c}function a(){const l=We.getState().selection.length>0,c=Yt.getState(),{machineState:u,machines:h}=Et.getState(),d=We.getState().layers.some(w=>w.workpieces.length>0),f=Lh({active:c.active,phase:c.phase,connected:u?.connected??!1,hasContent:d}),g=h.length>0,_=je.getState().showToolpaths,m=je.getState().showTravelMoves,p=je.getState().showWorkpieceImages,C=je.getState().showNogoZones,M=je.getState().showStock,x=[{id:"model",label:"Model",groups:()=>[{label:"Tools",items:[["select","pointer","Select"],["move","move","Move"],["rotate","rotate","Rotate"],["scale","scale","Scale"]].map(([w,T,L])=>({iconName:T,label:L,title:L,active:t===w,onClick:()=>{t=w,a(),n.setTool(w)}}))},{label:"Project",items:[{iconName:"filePlus",title:"New project (Ctrl+N)",label:"New",onClick:n.newDoc},{iconName:"folder",title:"Open project (Ctrl+O)",label:"Open",onClick:n.openProject},{iconName:"save",title:"Save project (Ctrl+S)",label:"Save",onClick:n.saveProject},{iconName:"upload",title:"Import file (Ctrl+I)",label:"Import",onClick:n.importFile},{iconName:"save",title:"Export G-code",label:"Export",onClick:n.exportGcode}]},{label:"Modify",items:[{iconName:"copy",title:"Duplicate selection (Ctrl+D)",label:"Duplicate",disabled:!l,onClick:n.duplicateSelected},{iconName:"trash",title:"Delete selection (Del)",label:"Delete",disabled:!l,onClick:n.deleteSelected},{iconName:"flipH",title:"Flip horizontally",label:"Flip H",disabled:!l,onClick:()=>n.flip("horizontal")},{iconName:"flipV",title:"Flip vertically",label:"Flip V",disabled:!l,onClick:()=>n.flip("vertical")}]},{label:"Arrange",items:[["left","alignLeft","Left"],["center_horizontal","alignCenterH","Center H"],["right","alignRight","Right"],["top","alignTop","Top"],["center_vertical","alignCenterV","Center V"],["bottom","alignBottom","Bottom"]].map(([w,T,L])=>({iconName:T,label:L,title:`Align ${L}`,disabled:!l,onClick:()=>n.align(w)}))},{label:"Layout",items:[{iconName:"grid",title:"Arrange all workpieces automatically",label:"Auto layout",disabled:!d,onClick:w=>n.autoLayout(w)}]}]},{id:"machine",label:"Machine",groups:()=>[{label:"Setup",items:[{iconName:"sliders",title:"Edit machine geometry and motion settings",label:"Machine settings",disabled:!g,onClick:()=>n.machineSettings()}]},{label:"Positioning",items:[{iconName:"home",title:"Home all axes",label:"Home all",disabled:!g,onClick:n.homeAll},{iconName:"crosshair",title:"Set work origin at the current position",label:"Set origin",disabled:!g,onClick:n.setOriginHere},{iconName:"crosshairSmall",title:"Move to the work origin",label:"Go to origin",disabled:!g,onClick:n.goToOrigin}]}]},{id:"job",label:"Job",groups:()=>[{label:"Job",items:[{iconName:"zap",title:"Add a step to the active layer",label:"Add step",disabled:!d,onClick:()=>n.addStep()},{iconName:"play",title:"Start job",label:"Start",disabled:!f.startEnabled,onClick:n.startJob},{iconName:"frame",title:"Frame workarea (Home)",label:"Frame",onClick:n.frameSelection}]},{label:"Output",items:[{iconName:"save",title:"Export G-code",label:"Export",onClick:n.exportGcode}]}]},{id:"view",label:"View",groups:()=>[{label:"Views",items:[{iconName:"viewTop",title:"Top view (1)",label:"Top",onClick:()=>n.viewPreset("top")},{iconName:"viewFront",title:"Front view (2)",label:"Front",onClick:()=>n.viewPreset("front")},{iconName:"viewIso",title:"Isometric view (3)",label:"Iso",onClick:()=>n.viewPreset("iso")},{iconName:"fit",title:"Zoom to fit (Home)",label:"Fit",onClick:()=>n.viewPreset("fit")}]},{label:"Show",items:[{iconName:"eye",title:"Toggle toolpaths",label:"Toolpaths",active:_,onClick:()=>Ah(!_)},{iconName:"eye",title:"Toggle travel moves",label:"Travels",active:m,onClick:()=>Ch(!m)},{iconName:"eye",title:"Toggle workpiece images",label:"Images",active:p,onClick:()=>Rh(!p)},{iconName:"eye",title:"Toggle no-go zones",label:"No-go zones",active:C,onClick:()=>Ph(!C)},{iconName:"eye",title:"Toggle stock material shapes",label:"Stock",active:M,onClick:()=>fp(!M)}]},{label:"Zoom",items:[{iconName:"zoomOut",title:"Zoom out",label:"Out",onClick:()=>n.zoom(120)},{iconName:"zoomIn",title:"Zoom in",label:"In",onClick:()=>n.zoom(-120)}]}]},{id:"help",label:"Help",groups:()=>[{label:"Help",items:[{iconName:"info",title:"About Rayforge",label:"About",onClick:n.about}]}]}],P=x.find(w=>w.id===i)??x[0];i=P.id,Cn(e),e.append(r(x),o(P))}return a(),je.subscribe(a),We.subscribe(a),Et.subscribe(a),Yt.subscribe(a),{root:e}}function Ue(n={}){const e=E("button","btn"),t=n.variant??(n.label?"ghost":"icon");if(e.classList.add(`btn-${t}`),n.iconName&&e.appendChild(Gt(n.iconName,n.iconSize??16)),n.label){const i=E("span","btn-label");i.textContent=n.label,e.appendChild(i)}return n.title&&(e.title=n.title),n.disabled&&(e.disabled=!0),n.active&&e.classList.add("active"),n.onClick&&e.addEventListener("click",n.onClick),e}function Dh(n){return n instanceof Error?n.message:String(n)}async function Ih(n,e){let t;try{t=await le.getStepTypes()}catch(d){_e("error",`Loading step types failed: ${Dh(d)}`);return}const i=E("div","step-picker"),s=E("div","panel-note");s.textContent=`Add a step to layer “${n.name}”`,i.appendChild(s);for(const d of t){const f=Ue({label:d.typelabel||d.step_type,iconName:"zap",onClick:()=>{h(),Sp(n,d.step_type,e)}});f.classList.add("step-picker-item"),i.appendChild(f)}const r=E("div","modal-backdrop"),o=E("div","dialog"),a=E("header","dialog-header"),l=E("span","dialog-title");l.textContent="Add step";const c=Ue({iconName:"x",variant:"icon",title:"Close",onClick:h});a.append(l,c);const u=E("div","dialog-body");u.appendChild(i),o.append(a,u),r.appendChild(o),r.addEventListener("click",d=>{d.target===r&&h()}),document.body.appendChild(r);function h(){r.remove()}}function Nh(n,e){return Ih({uid:n.uid,name:n.name},e)}async function Sp(n,e,t){try{await le.postStep(n.uid,{step_type:e}),await t()}catch(i){_e("error",`Adding step failed: ${Dh(i)}`)}}let zs=null;function so(){zs?.remove(),zs=null}document.addEventListener("pointerdown",n=>{zs&&!zs.contains(n.target)&&so()});document.addEventListener("keydown",n=>{n.key==="Escape"&&so()});function ar(n,e,t){so();const i=E("div","menu-popup context-popup");for(const s of t){if(s.separator){i.appendChild(E("div","menu-sep"));continue}const r=E("button","menu-item");r.type="button",s.danger&&r.classList.add("danger");const o=E("span","menu-item-label");o.textContent=s.label??"",r.appendChild(o),s.disabled?r.disabled=!0:r.addEventListener("click",()=>{so(),s.action?.()}),i.appendChild(r)}i.style.left=`${Math.max(4,Math.min(n,window.innerWidth-230))}px`,i.style.top=`${Math.max(4,Math.min(e,window.innerHeight-40-t.length*26))}px`,document.body.appendChild(i),zs=i}function bp(n){const e=E("div","tree");for(const t of n)e.appendChild(Mp(t));if(n.length===0){const t=E("div","tree-empty");t.textContent="Empty document",e.appendChild(t)}return e}function Mp(n){const e=E("div","tree-row");e.style.paddingLeft=`${8+n.depth*14}px`,e.dataset.uid=n.id,n.selected&&e.classList.add("selected"),n.active&&e.classList.add("active-row"),n.dimmed&&e.classList.add("dimmed");const t=E("span","tree-expander");if(n.expandable){const s=E("button","tree-expander-btn");s.type="button",s.appendChild(Gt(n.expanded?"chevronDown":"chevronRight",12)),s.addEventListener("click",r=>{r.stopPropagation(),n.onToggleExpanded?.()}),t.appendChild(s)}if(e.appendChild(t),n.visible!==void 0){const s=E("button","tree-vis");s.type="button",s.title="Toggle visibility",s.appendChild(Gt(n.visible?"eye":"eyeOff",13)),s.classList.toggle("off",!n.visible),s.addEventListener("click",r=>{r.stopPropagation(),n.onToggleVisible?.()}),e.appendChild(s)}if(n.colorChip)if(n.onChipClick){const s=E("button","tree-chip-btn");s.type="button",s.title="Change color",s.appendChild(wl(n.colorChip)),s.addEventListener("click",r=>{r.stopPropagation(),n.onChipClick?.(r)}),e.appendChild(s)}else e.appendChild(wl(n.colorChip));if(n.icon&&e.appendChild(Gt(n.icon,13)),n.statusDot){const s=E("span",`tree-dot ${n.statusDot}`);e.appendChild(s)}const i=E("span","tree-label");if(i.textContent=n.label,e.appendChild(i),n.renamable&&n.onRename&&i.addEventListener("dblclick",s=>{s.stopPropagation(),wp(i,n)}),n.hint){const s=E("span","tree-hint");s.textContent=n.hint,e.appendChild(s)}for(const s of n.trailing??[]){const r=E("span","tree-actions");r.appendChild(s),e.appendChild(r)}return e.addEventListener("click",s=>n.onSelect?.(s)),n.onContext&&e.addEventListener("contextmenu",s=>{s.preventDefault(),n.onContext?.(s)}),e}function wl(n){const e=E("span","tree-chip");return e.style.background=n,e}function wp(n,e){const t=E("input","tree-rename-input");t.type="text",t.value=e.label,t.spellcheck=!1;let i=!1;const s=()=>{if(i)return;const r=t.value.trim();r&&r!==e.label&&e.onRename?.(r)};t.addEventListener("click",r=>r.stopPropagation()),t.addEventListener("keydown",r=>{r.key==="Enter"?(r.preventDefault(),s(),i=!0,t.blur()):r.key==="Escape"&&(i=!0,t.blur())}),t.addEventListener("blur",s),n.replaceChildren(t),t.focus(),t.select()}function Ep(n,e){Cn(n),n.appendChild(bp(e))}function Tp(n){const e=E("div","layers-panel"),t=E("div","layers-header"),i=E("span","layers-title");i.textContent="Layers";const s=Ue({iconName:"filePlus",iconSize:14,variant:"icon",title:"Add layer",onClick:()=>{Ap(n)}});t.append(i,s),e.appendChild(t);const r=E("div","layers-tree");e.appendChild(r);const o=()=>{je.getState().activeLeftTab==="document"&&Ep(r,zp(We.getState(),n))};return We.subscribe(o),je.subscribe(o),o(),{root:e}}function Dn(n){return n instanceof Error?n.message:String(n)}async function Ap(n){try{await le.postLayer(),await n.refreshDoc()}catch(e){_e("error",`Adding layer failed: ${Dn(e)}`)}}async function Cp(n,e){try{await le.deleteLayer(n.uid),await e.refreshDoc()}catch(t){_e("error",`Deleting layer failed: ${Dn(t)}`)}}async function Rp(n,e){if(!n.is_active){Lf(n.uid);try{await le.activateLayer(n.uid)}catch(t){_e("error",`Activating layer failed: ${Dn(t)}`),await e.refreshDoc()}}}async function Pp(n,e){const t=n.name;En(n.uid,{name:e});try{await le.putLayer(n.uid,{name:e})}catch(i){En(n.uid,{name:t}),_e("error",`Renaming failed: ${Dn(i)}`)}}async function Lp(n,e){const t=n.color;if(e!==t){En(n.uid,{color:e});try{await le.putLayer(n.uid,{color:e})}catch(i){En(n.uid,{color:t}),_e("error",`Color change failed: ${Dn(i)}`)}}}function Dp(n){const e=!n.visible;En(n.uid,{visible:e}),le.putLayer(n.uid,{visible:e}).catch(t=>{En(n.uid,{visible:!e}),_e("error",`Visibility change failed: ${Dn(t)}`)})}async function ro(n,e,t){const i=We.getState(),s=i.layers.findIndex(l=>l.uid===n.uid),r=s+e;if(s<0||r<0||r>=i.layers.length)return;const o=i.layers.map(l=>l.uid),[a]=o.splice(s,1);o.splice(r,0,a),Df(o);try{await le.reorderLayers(o)}catch(l){_e("error",`Reordering failed: ${Dn(l)}`),await t.refreshDoc()}}function El(n,e){const t=We.getState().layers.length<=1;return[{label:"Rename",action:()=>Ip(n.uid)},{label:"Move up",action:()=>{ro(n,-1,e)}},{label:"Move down",action:()=>{ro(n,1,e)}},{separator:!0},{label:"Add step…",action:()=>{Nh(n,e.refreshDoc)}},{label:"Delete layer",danger:!0,disabled:t,action:()=>{Cp(n,e)}}]}function Ip(n){document.querySelector(`.tree-row[data-uid="${n}"] .tree-label`)?.dispatchEvent(new MouseEvent("dblclick"))}function Tl(n,e){return[{label:"Duplicate",action:()=>{Np(n,e)}},{label:"Delete",danger:!0,action:()=>{Up([n.uid],e)}}]}async function Np(n,e){try{await le.duplicateItems([n.uid]),await e.refreshDoc()}catch(t){_e("error",`Duplicating failed: ${Dn(t)}`)}}async function Up(n,e){try{await le.deleteItems(n),await e.refreshDoc()}catch(t){_e("error",`Deleting failed: ${Dn(t)}`)}}async function Op(n,e){try{await le.deleteStep(n.uid),await e.refreshDoc()}catch(t){_e("error",`Deleting step failed: ${Dn(t)}`)}}function xs(n,e,t){const i=E("button","tree-btn");return i.type="button",i.title=e,i.appendChild(Gt(n,12)),i.addEventListener("click",s=>{s.stopPropagation(),t(s)}),i}function Fp(n,e,t,i){const s=E("input","color-popover");s.type="color",s.value=kp(t),s.addEventListener("input",()=>i(s.value)),s.addEventListener("change",()=>{i(s.value),window.setTimeout(()=>s.remove(),50)}),s.addEventListener("blur",()=>{window.setTimeout(()=>s.remove(),50)}),s.style.left=`${n}px`,s.style.top=`${e}px`,document.body.appendChild(s),s.click()}function kp(n){return/^#[0-9a-fA-F]{6}$/.test(n)?n:/^#[0-9a-fA-F]{3}$/.test(n)?`#${n[1]}${n[1]}${n[2]}${n[2]}${n[3]}${n[3]}`:"#4d9fff"}function zp(n,e){const t=[],i=n.layers.length;return n.layers.forEach((s,r)=>{const o=Pf(n,s.uid),a={id:s.uid,depth:0,label:s.name,colorChip:s.color,expandable:!0,expanded:o,selected:n.selection.includes(s.uid),visible:s.visible,active:s.is_active,renamable:!0,trailing:[xs("chevronUp","Move layer up",()=>{ro(s,-1,e)}),xs("chevronDown","Move layer down",()=>{ro(s,1,e)}),xs("kebab","Layer actions",l=>{ar(l.clientX,l.clientY,El(s,e))})],onToggleExpanded:()=>Rf(s.uid),onSelect:l=>{ei([s.uid],l.shiftKey),Rp(s,e)},onToggleVisible:()=>Dp(s),onChipClick:l=>{Fp(l.clientX,l.clientY,s.color,c=>{Lp(s,c)})},onRename:l=>{Pp(s,l)},onContext:l=>{ei([s.uid]),ar(l.clientX,l.clientY,El(s,e))}};if(r===0?a.trailing=a.trailing?.slice(1):r===i-1&&(a.trailing=a.trailing?.slice(0,1)),t.push(a),!!o){for(const l of s.workpieces)t.push({id:l.uid,depth:1,label:l.name,icon:"box",colorChip:s.color,dimmed:!s.visible,selected:n.selection.includes(l.uid),trailing:[xs("kebab","Workpiece actions",c=>{ar(c.clientX,c.clientY,Tl(l,e))})],onSelect:c=>ei([l.uid],c.shiftKey),onContext:c=>{ei([l.uid]),ar(c.clientX,c.clientY,Tl(l,e))}});for(const l of s.steps)t.push({id:l.uid,depth:1,label:l.name,icon:"zap",hint:l.typelabel??l.step_type,statusDot:l.visible===void 0?l.enabled?"on":"off":l.visible?"on":"off",dimmed:!s.visible,selected:n.selection.includes(l.uid),trailing:[xs("trash","Delete step",()=>{Op(l,e)})],onSelect:c=>ei([l.uid],c.shiftKey)})}}),t}function Uh(n,e,t,i){const s=E("aside",`${n}-panel floating-panel`),r=E("span","floating-card-title");r.textContent=e;const o=E("button","btn-icon btn-ghost panel-collapse");o.type="button",o.title=`Collapse ${e.toLowerCase()} panel`,o.appendChild(Gt(n==="left"?"chevronLeft":"chevronRight",15));const a=E("header","floating-card-header");a.append(r,o),s.append(a,i);const l=E("span","panel-stub-label");l.textContent=e;const c=E("button",`panel-stub ${n}-stub`);return c.type="button",c.title=`Expand ${e.toLowerCase()} panel`,c.append(Gt(t,15),l),o.addEventListener("click",()=>{s.hidden=!0,c.hidden=!1}),c.addEventListener("click",()=>{c.hidden=!0,s.hidden=!1}),c.hidden=!0,{root:s,stub:c}}function _o(n,e,t){const i=E("div","tabs"),s=new Map;for(const o of n){const a=E("button","tab");a.type="button",a.textContent=o.label,a.dataset.tab=o.id,a.addEventListener("click",()=>t(o.id)),s.set(o.id,a),i.appendChild(a)}function r(o){for(const[a,l]of s)l.classList.toggle("active",a===o)}return r(e),{root:i,setActive:r}}function Bp(n){const e=E("div","panel-tabs"),t=E("div","panel-content"),i=Tp(n),s=E("div","doc-tree-host");s.appendChild(i.root);const r=E("div","machine-host");let o=null,a=je.getState().activeLeftTab;function l(d){a=d,Th(d),o?.setActive(d),Cn(t),t.appendChild(d==="machine"?r:s)}o=_o([{id:"document",label:"Document"},{id:"machine",label:"Machine"}],je.getState().activeLeftTab,l),e.appendChild(o.root);const c=E("div","floating-panel-body");c.append(e,t);const u=Uh("left","Design","layers",c),h=()=>{Cn(r),r.appendChild(Hp())};return Et.subscribe(h),je.subscribe(d=>{if(d.activeLeftTab!==a){l(d.activeLeftTab);return}d.activeLeftTab==="machine"&&h()}),l(je.getState().activeLeftTab),{root:u.root,stub:u.stub,docContent:s,machineContent:r}}function Po(n){return typeof n=="number"&&Number.isFinite(n)?n.toFixed(1):"--"}function Lo(n){return n?`${Po(n.x)}, ${Po(n.y)}, ${Po(n.z)}`:"--, --, --"}function Hp(){const{machines:n,machineState:e,activeMachineId:t,viewport:i}=Et.getState(),s=E("div","machine-content"),r=n.find(c=>c.id===t);if(n.length>0){const c=E("div","machine-list");c.title=n.length===1?"The web API controls the active machine; this list is display-only.":"The web API controls the active machine (marked); selection is display-only.";for(const u of n){const h=E("div","machine-row");h.classList.toggle("active-machine",u.is_active);const d=E("div","machine-row-info"),f=E("span","machine-name");f.textContent=u.name;const g=E("span","machine-driver");g.textContent=u.driver||"no driver",d.append(f,g);const _=E("span",`machine-pill ${u.connected?"on":"off"}`);_.textContent=u.connected?"Connected":"Offline",h.append(d,_),c.appendChild(h)}s.appendChild(c)}const o=E("div","machine-info"),a=e?.active_wcs??i?.active_wcs.name??"--",l=[["Driver",r?.driver||"--"],["Connection",e?e.connected?"Connected":"Offline":"--"],["Status",e?.status??"--"],["Machine pos",Lo(e?.machine_pos??null)],["Work pos",Lo(e?.work_pos??null)],["WCO",Lo(e?.wco??null)],["Active WCS",a]];e?.feed_rate!==null&&e?.feed_rate!==void 0&&l.push(["Feed",`${e.feed_rate}`]),e?.spindle_speed!==null&&e?.spindle_speed!==void 0&&l.push(["Spindle",`${e.spindle_speed}`]);for(const[c,u]of l){const h=E("div","kv-row"),d=E("span","kv-key");d.textContent=c;const f=E("span","kv-value mono");f.textContent=u,h.append(d,f),o.appendChild(h)}return s.appendChild(o),s}const Gp={switch:"toggle",slider:"slider",choice:"select",textarea:"textarea",input:"number",length:"number",speed:"number",angle:"number"},Vp={mm:"mm","mm/min":"mm/min",deg:"°"};function Wp(n){return Gp[n]??"number"}function Xp(n){const e=n.min??0,t=n.max??100;return n.step??(t-e)/100}function Al(n){return n==="boolean"?!1:n==="number"?0:""}function $p(n,e){if(n.key in e&&e[n.key]!==void 0&&e[n.key]!==null){const t=e[n.key];if(n.type==="number"&&typeof t=="string"){const i=Number.parseFloat(t);return Number.isNaN(i)?Al("number"):i}return n.type==="boolean"?!!t:t}return n.default!==void 0&&n.default!==null?n.default:Al(n.type)}function Yp(n){return String(n)}function qp(n,e){const i=Array.isArray(n.options)&&n.options.length>0?"select":n.type==="boolean"?"toggle":n.kind==="input"&&n.type==="string"?"text":Wp(n.kind);return{key:n.key,widget:i,kind:n.kind,type:n.type,label:n.label,description:n.description??null,value:$p(n,e),min:n.min??null,max:n.max??null,step:n.step??null,unit:n.unit?Vp[n.unit]??n.unit:null,options:(n.options??[]).map(s=>({value:Yp(s.value),label:s.label})),optional:!!n.optional}}function Rc(n,e){return n.groups.map(t=>({title:t.title,fields:t.fields.filter(i=>i.key!=="").map(i=>qp(i,e))}))}function jp(n,e,t){return{type:"TraceSpec",threshold:Math.min(1,Math.max(0,n/255)),auto_threshold:!1,invert:e,ppi:t}}function Zp(){return{type:"PassthroughSpec"}}function un(n){const e=E("label","num-field"),t=E("span","num-field-label");t.textContent=n.label;const i=E("input","num-field-input");i.type="text",i.inputMode="decimal",i.spellcheck=!1,i.value=ys(n.value),n.disabled&&(i.disabled=!0);const s=()=>{const o=Number.parseFloat(i.value.replace(",","."));if(Number.isNaN(o)){i.value=ys(n.value);return}let a=o;n.min!==void 0&&(a=Math.max(n.min,a)),n.max!==void 0&&(a=Math.min(n.max,a)),i.value=ys(a),a!==n.value&&(n.value=a,n.onCommit?.(a))};i.addEventListener("keydown",o=>{o.key==="Enter"?(o.preventDefault(),s(),i.blur()):o.key==="Escape"&&(i.value=ys(n.value),i.blur())}),i.addEventListener("blur",s),e.append(t);const r=E("span","num-field-wrap");if(r.appendChild(i),n.unit){const o=E("span","num-field-unit");o.textContent=n.unit,r.appendChild(o)}return e.appendChild(r),{root:e,setValue(o){document.activeElement!==i&&(i.value=ys(o)),n.value=o}}}function ys(n){return String(Math.round(n*100)/100)}function _n(n,e){const t=E("section","panel"),i=E("header","panel-header"),s=E("span","panel-title");s.textContent=n,i.appendChild(s),e&&i.appendChild(e);const r=E("div","panel-body");return t.append(i,r),{root:t,header:i,body:r}}function rs(n,e,t,i){const s=E("label","select-field"),r=E("span","num-field-label");r.textContent=n;const o=E("select","select-input");for(const a of e){const l=E("option");l.value=a.value,l.textContent=a.label,o.appendChild(l)}return o.value=t,i&&o.addEventListener("change",()=>i(o.value)),s.append(r,o),{root:s,setValue(a){o.value=a},setDisabled(a){o.disabled=a}}}function Oh(n,e,t,i,s,r){const o=E("label","slider-field"),a=E("span","num-field-label");a.textContent=n;const l=E("input","slider-input");l.type="range",l.min=String(e),l.max=String(t),l.step=String(i),l.value=String(s);const c=E("span","slider-readout");return c.textContent=String(s),l.addEventListener("input",()=>{c.textContent=l.value,r?.(Number.parseFloat(l.value))}),o.append(a,l,c),{root:o,setValue(u){l.value=String(u),c.textContent=String(u)}}}function bi(n){const e=E("label","num-field"),t=E("span","num-field-label");t.textContent=n.label;const i=E("input","num-field-input text-field-input");i.type="text",i.value=n.value,i.spellcheck=!1,n.placeholder&&(i.placeholder=n.placeholder);const s=()=>{const r=i.value.trim();r!==n.value&&(n.value=r,n.onCommit?.(r))};return i.addEventListener("keydown",r=>{r.key==="Enter"?(r.preventDefault(),s(),i.blur()):r.key==="Escape"&&(i.value=n.value,i.blur())}),i.addEventListener("blur",s),e.append(t,i),{root:e,setValue(r){document.activeElement!==i&&(i.value=r),n.value=r}}}function os(n,e){const t=E("button","toggle");t.type="button",t.role="switch",t.setAttribute("aria-checked",String(n)),n&&t.classList.add("on");const i=E("span","toggle-knob");return t.appendChild(i),t.addEventListener("click",()=>{const s=!t.classList.contains("on");t.classList.toggle("on",s),t.setAttribute("aria-checked",String(s)),e?.(s)}),{root:t,setChecked(s){t.classList.toggle("on",s),t.setAttribute("aria-checked",String(s))}}}const Jp=300,Cl=new Map;function Kp(n,e){const t=E("div","step-form"),i=E("span","saved-hint"),s=Ue({iconName:"trash",variant:"icon",title:"Delete step",onClick:()=>{tm(n,e)}}),r=_n(n.name,s);r.header.appendChild(i);const o=E("div","panel-note");o.textContent=`Type: ${n.typelabel??n.step_type}`,r.body.appendChild(o);const a=E("div","step-form-body");r.body.appendChild(a),t.appendChild(r.root);const l=Cl.get(n.uid);return l?Rl(a,n,l,i):(a.appendChild(Qp()),le.getStepSchema(n.uid).then(c=>{Cl.set(n.uid,c),a.isConnected&&(Cn(a),Rl(a,n,c,i))}).catch(c=>{a.isConnected&&(Cn(a),a.appendChild(em(c)))})),t}function Qp(){const n=E("div","panel-note");return n.textContent="Loading schema…",n}function em(n){const e=E("div","panel-note");return e.textContent=`Schema unavailable: ${Pc(n)}`,e}function Pc(n){return n instanceof Error?n.message:String(n)}async function tm(n,e){try{await le.deleteStep(n.uid),await e.refreshDoc()}catch(t){_e("error",`Deleting step failed: ${Pc(t)}`)}}function Rl(n,e,t,i){const s={...e.params},r=new Set;let o=null,a=!1;const l=(g,_)=>{i.textContent=g,i.className=`saved-hint ${_}`},c=async()=>{if(o=null,r.size===0||a)return;const g={};for(const _ of r)g[_]=s[_];r.clear(),a=!0,l("Saving…","saving");try{await le.putStep(e.uid,{params:g}),l("Saved","done"),window.setTimeout(()=>{i.textContent==="Saved"&&l("","")},1600)}catch(_){l("",""),_e("error",`Saving step failed: ${Pc(_)}`)}finally{a=!1,r.size>0&&o===null&&u()}},u=()=>{l("Editing…","dirty"),o!==null&&window.clearTimeout(o),o=window.setTimeout(()=>{c()},Jp)},h=(g,_)=>{s[g]=_,r.add(g),u()},d=Rc(t,e.params);for(const g of d){if(g.fields.length===0)continue;const _=E("div","schema-group");if(g.title){const m=E("div","schema-group-title");m.textContent=g.title,_.appendChild(m)}for(const m of g.fields)_.appendChild(Lc(m,h));n.appendChild(_)}const f=e.workpiece_uids;f.length>0&&n.appendChild(im(e,f))}function Lc(n,e){const t=E("div","schema-field");switch(n.description&&(t.title=n.description),n.widget){case"toggle":{const i=os(!!n.value,o=>e(n.key,o)),s=E("div","schema-field-row"),r=E("span","num-field-label");return r.textContent=n.label,s.append(r,i.root),t.appendChild(s),t}case"slider":{const i=n.min??0,s=n.max??100,r=Oh(n.label,i,s,Xp(n),Number(n.value)||0,o=>e(n.key,o));return t.appendChild(r.root),t}case"select":{const i=n.options.find(r=>r.value===String(n.value))?.value??n.options[0]?.value??"",s=rs(n.label,n.options,i,r=>e(n.key,nm(n,r)));return s.root.classList.add("in-panel"),t.appendChild(s.root),t}case"textarea":{const i=E("span","num-field-label");i.textContent=n.label;const s=E("textarea","textarea-input");return s.value=String(n.value??""),s.spellcheck=!1,s.addEventListener("input",()=>e(n.key,s.value)),t.append(i,s),t}case"text":{const i=bi({label:n.label,value:String(n.value??""),onCommit:s=>e(n.key,s)});return t.appendChild(i.root),t}default:{const i=Number(n.value),s=un({label:n.label,value:Number.isNaN(i)?0:i,unit:n.unit??void 0,min:n.min??void 0,max:n.max??void 0,step:n.step??void 0,onCommit:r=>e(n.key,r)});return t.appendChild(s.root),t}}}function nm(n,e){if(n.type==="number"){const t=Number(e);return Number.isNaN(t)?e:t}return e}function im(n,e){const t=E("div","wp-filter"),i=E("span","num-field-label");i.textContent="Applied to workpieces",t.appendChild(i);const s=e.map(a=>sm(n.uid)?.workpieces.find(c=>c.uid===a)?.name??a),r=E("div","chip-list");for(const a of s){const l=E("span","chip");l.textContent=a,r.appendChild(l)}t.appendChild(r);const o=E("div","panel-note");return o.textContent="Read-only: this filter is managed by the desktop editor",t.appendChild(o),t}function sm(n){for(const e of We.getState().layers)if(e.steps.some(t=>t.uid===n))return e;return null}class rm{items=[];limit;index=null;draft="";constructor(e=100){this.limit=e}get size(){return this.items.length}push(e){const t=e.trim();if(t){if(this.items[this.items.length-1]===t){this.index=null,this.draft="";return}this.items.push(t),this.items.length>this.limit&&this.items.shift(),this.index=null,this.draft=""}}older(e){return this.items.length===0?null:(this.index===null&&(this.draft=e,this.index=this.items.length),this.index>0&&this.index--,this.items[this.index]??null)}newer(){return this.index===null?null:this.index<this.items.length-1?(this.index++,this.items[this.index]??null):(this.index=null,this.draft)}reset(){this.index=null,this.draft=""}}let om=0;class am{lines=[];limit;constructor(e=400){this.limit=e}all(){return[...this.lines]}append(e,t){return this.lines.push({id:++om,kind:e,text:t}),this.lines.length>this.limit&&this.lines.splice(0,this.lines.length-this.limit),this.all()}}function cm(n){return n instanceof Error?n.message:String(n)}function lm(){const n=E("div","console-section"),e=_n("G-code console"),t=new rm,i=new am,s=E("div","console-log mono"),r=u=>{for(;s.firstChild;)s.removeChild(s.firstChild);for(const h of u){const d=E("div",`console-line console-${h.kind}`);d.textContent=h.text,s.appendChild(d)}s.scrollTop=s.scrollHeight};i.append("info","Raw G-code is sent to the machine; responses are not streamed."),r(i.all());const o=E("div","console-input-row"),a=E("input","console-input mono");a.type="text",a.spellcheck=!1,a.placeholder="e.g. G0 X10 Y10";const l=()=>{const u=a.value.trim();u&&(a.value="",t.push(u),r(i.append("cmd",`> ${u}`)),le.runRaw(u).then(()=>r(i.append("ok","sent"))).catch(h=>r(i.append("error",`error: ${cm(h)}`))))};a.addEventListener("keydown",u=>{if(u.key==="Enter")u.preventDefault(),l(),t.reset();else if(u.key==="ArrowUp"){u.preventDefault();const h=t.older(a.value);h!==null&&(a.value=h)}else if(u.key==="ArrowDown"){u.preventDefault();const h=t.newer();h!==null&&(a.value=h)}});const c=Ue({iconName:"terminal",title:"Send G-code",onClick:()=>{l(),t.reset()}});return o.append(a,c),e.body.append(s,o),n.appendChild(e.root),{root:n}}function cr(n,e,t,i,s){return{direction:`${e?"+":"-"}${n}`,screen:t,axis:n,towardOrigin:!e,nativeAxis:s==="native"?n:n==="x"?"y":"x",nativeNegative:um(n,e,i)}}function um(n,e,t){const i=n==="x"?t.x_axis_right:t.y_axis_down,s=n==="x"?t.x_axis_negative:t.y_axis_negative;let r=e?1:-1;return s&&(r=-r),i&&(r=-r),r<0}function hm(n,e="native"){return{right:cr("x",!n.x_axis_right,"right",n,e),left:cr("x",n.x_axis_right,"left",n,e),up:cr("y",!n.y_axis_down,"up",n,e),down:cr("y",n.y_axis_down,"down",n,e)}}function dm(n){const e=`${n.nativeAxis.toUpperCase()}${n.nativeNegative?"-":"+"}`,t=n.towardOrigin?"toward origin":"away from origin";return`Jog ${n.screen} (panel ${n.direction}, machine ${e}, ${t})`}const Pl=["G54","G55","G56","G57","G58","G59"],fm=[1,5,10];function pm(n){return n instanceof Error?n.message:String(n)}const mm={up:"chevronUp",down:"chevronDown",left:"chevronLeft",right:"chevronRight"};function gm(){const n=E("div","control-section");let e=10,t=0,i=5,s="";const r=(H,D)=>{H().catch(W=>_e("error",`${D}: ${pm(W)}`))},o=H=>{r(()=>le.jog(H,e,t>0?t:void 0),"Jog failed")},a=_n("Jog"),l=E("div","jog-grid"),c=E("div","seg-row"),u=new Map;for(const H of fm){const D=E("button","seg-btn");D.type="button",D.textContent=`${H}`,D.title=`Jog distance ${H} mm`,D.addEventListener("click",()=>{e=H;for(const[W,Q]of u)Q.classList.toggle("active",W===e)}),u.set(H,D),c.appendChild(D)}u.get(e).classList.add("active");const h=un({label:"Speed mm/min (0 = driver default)",value:0,min:0,step:100,onCommit:H=>{t=H}});a.body.append(l,c,h.root),n.appendChild(a.root);const d=[];function f(H,D,W){const Q=E("button","btn jog-btn");Q.type="button",Q.title=H,Q.appendChild(Gt(D,14));const me=E("span","jog-axis");return me.textContent=W,Q.appendChild(me),l.appendChild(Q),d.push(Q),Q}function g(){const{viewport:H}=Et.getState(),D=H?.panel_orientation??"native",W=hm({x_axis_right:H?.x_axis_right??!1,y_axis_down:H?.y_axis_down??!1,x_axis_negative:H?.x_axis_negative??!1,y_axis_negative:H?.y_axis_negative??!1},D),Q=[W.right.direction,W.up.direction,H?.x_axis_negative,H?.y_axis_negative].join(",");if(Q===s)return;s=Q;for(const te of[...l.children])l.removeChild(te);d.length=0;const me=[[W.up,"2",1],[W.left,"1",2],[W.right,"3",2],[W.down,"2",3]];for(const[te,xe,Fe]of me){const Te=te.direction.startsWith("+")?"+":"-",Je=f(dm(te),mm[te.screen]??"move",`${te.axis.toUpperCase()}${Te}`);Je.style.gridColumn=xe,Je.style.gridRow=String(Fe),Je.addEventListener("click",()=>o(te.direction))}const De=f("Jog up (panel +z)","chevronUp","Z+");De.style.gridColumn="3",De.style.gridRow="1",De.addEventListener("click",()=>o("+z"));const Ie=f("Jog down (panel -z)","chevronDown","Z-");Ie.style.gridColumn="3",Ie.style.gridRow="3",Ie.addEventListener("click",()=>o("-z"));const j=f("Home all axes","home","");j.classList.add("jog-home"),j.style.gridColumn="2",j.style.gridRow="2",j.addEventListener("click",()=>r(()=>le.homeMachine(),"Homing failed"))}const _=_n("Homing"),m=E("div","btn-row"),p=Ue({label:"Home all",onClick:()=>r(()=>le.homeMachine(),"Homing failed")});m.appendChild(p);const C=["x","y","z"].map(H=>Ue({label:`Home ${H.toUpperCase()}`,onClick:()=>r(()=>le.homeMachine([H]),"Homing failed")}));for(const H of C)m.appendChild(H);_.body.appendChild(m),n.appendChild(_.root);const M=_n("Work origin"),x=E("div","btn-row"),P=Ue({label:"Set origin here",title:"Zero the active WCS at the current position",onClick:()=>r(()=>le.setWorkOriginHere(),"Set work origin failed")}),w=Ue({label:"Go to origin",title:"Move to the work-area origin",onClick:()=>{const H=Et.getState().viewport?.workarea_origin_panel;if(!H){_e("warning","Work-area origin unknown");return}r(()=>le.moveTo(H.x,H.y),"Move failed")}});x.append(P,w);const T=E("div","wcs-row"),L=rs("Coordinate system",Pl.map(H=>({value:H,label:H})),"G54",H=>r(()=>le.selectWcs(H),"Select WCS failed")),y=Ue({label:"Sync",title:"Re-read WCS offsets from the device",onClick:()=>r(()=>le.syncWcs(),"WCS sync failed")});T.append(L.root,y),M.body.append(x,T),n.appendChild(M.root);const b=_n("Frame & laser"),N=Ue({label:"Frame job",iconName:"frame",title:"Trace the job bounding box without firing",onClick:()=>r(()=>le.frame(),"Framing failed")}),O=un({label:"Power %",value:i,min:0,max:100,onCommit:H=>{i=H}}),G=Ue({label:"Set power",onClick:()=>r(()=>le.setPower(i/100),"Set power failed")}),V=Ue({label:"Focus power",onClick:()=>r(()=>le.setFocusPower(i/100),"Set focus power failed")}),X=E("button","btn btn-danger fire-btn");X.type="button",X.textContent="TEST FIRE",X.title="Hold to fire the laser at the set power";const q=H=>{r(()=>le.setPower(H?i/100:0),"Laser test failed")};X.addEventListener("pointerdown",()=>q(!0));for(const H of["pointerup","pointerleave","pointercancel"])X.addEventListener(H,()=>q(!1));const J=E("div","btn-row");J.append(G,V,X),b.body.append(N,O.root,J),n.appendChild(b.root);function z(){const{machineState:H,viewport:D}=Et.getState(),W=H?.connected??!1,Q=Yt.getState().active,me=W&&!Q;g();for(const Ie of d)Ie.disabled=!me;p.disabled=!me;for(const Ie of C)Ie.disabled=!me;P.disabled=!me,w.disabled=!me,y.disabled=!W,N.disabled=!W||Q,L.setDisabled(!W),G.disabled=!W,V.disabled=!W,X.disabled=!W;const De=H?.active_wcs??D?.active_wcs.name??"G54";Pl.includes(De)&&L.setValue(De)}return Et.subscribe(z),Yt.subscribe(z),z(),{root:n}}let Dc=null;function _m(n){const e=E("div","panel-content"),t=E("div","right-panel-bottom"),i=E("div","panel-tabs"),s=E("div","panel-content machine-panel-content"),r=gm(),o=lm();let a=null;function l(h){a?.setActive(h),Cn(s),s.appendChild(h==="console"?o.root:r.root)}a=_o([{id:"control",label:"Control"},{id:"console",label:"Console"}],"control",l),i.appendChild(a.root),l("control"),t.append(i,s);const c=E("div","floating-panel-body");c.append(e,t);const u=Uh("right","Properties","sliders",c);return We.subscribe(h=>Ll(e,h,n)),Ll(e,We.getState(),n),Ac.subscribe(h=>{const d=Dc;if(!d)return;const f=h.transformPreview;if(!f){d.fields.x.setValue(d.doc.x),d.fields.y.setValue(d.doc.y),d.fields.w.setValue(d.doc.w),d.fields.h.setValue(d.doc.h),d.fields.a.setValue(d.doc.a);return}d.uid===f.uid&&(d.fields.x.setValue(f.pos.x),d.fields.y.setValue(f.pos.y),d.fields.w.setValue(f.size.w),d.fields.h.setValue(f.size.h),d.fields.a.setValue(f.angleDeg))}),{root:u.root,stub:u.stub,content:e,machineContent:s}}function Ll(n,e,t){Cn(n),Dc=null;const i=Ef(e.layers,e.selection);if(i.length===1){n.appendChild(vm(i[0],t));return}if(i.length>1){n.appendChild(ym(i.length,t));return}const s=e.selection.map(o=>wf(e.layers,o)).find(o=>o!==void 0);if(s){n.appendChild(Kp(s.step,t));return}const r=e.selection.map(o=>Mf(e.layers,o)).find(o=>o!==void 0);if(r){n.appendChild(bm(r,t));return}n.appendChild(wm())}function lr(n){return n instanceof Error?n.message:String(n)}function vm(n,e){const t=_n("Transform"),i=(u,h,d,f,g)=>{le.putItemTransform(n.uid,{pos_mm:{x:u,y:h},size_mm:{w:d,h:f},angle_deg:g}).then(()=>e.refreshDoc()).catch(_=>{_e("error",`Transform failed: ${String(_)}`)})},s={},r=(u,h,d,f)=>{const g=un({label:u,value:h,unit:f,onCommit:()=>{i(s.x?Ss(s.x,n.pos_mm.x):n.pos_mm.x,s.y?Ss(s.y,n.pos_mm.y):n.pos_mm.y,s.w?Ss(s.w,n.size_mm.w):n.size_mm.w,s.h?Ss(s.h,n.size_mm.h):n.size_mm.h,s.a?Ss(s.a,n.angle_deg):n.angle_deg)}});return s[d]=g,g.root},o=E("div","field-grid");o.append(r("X",n.pos_mm.x,"x","mm"),r("Y",n.pos_mm.y,"y","mm"),r("W",n.size_mm.w,"w","mm"),r("H",n.size_mm.h,"h","mm"),r("Angle",n.angle_deg,"a","°")),t.body.appendChild(o),t.body.appendChild(Fh(e)),Dc={uid:n.uid,doc:{x:n.pos_mm.x,y:n.pos_mm.y,w:n.size_mm.w,h:n.size_mm.h,a:n.angle_deg},fields:{x:s.x,y:s.y,w:s.w,h:s.h,a:s.a}};const a=_n("Workpiece"),l=[["Name",n.name],["Revisions",`geo ${n.geometry_revision} · tf ${n.transform_revision}`]];for(const[u,h]of l){const d=E("div","kv-row"),f=E("span","kv-key");f.textContent=u;const g=E("span","kv-value");g.textContent=h,d.append(f,g),a.body.appendChild(d)}const c=E("div");return c.append(t.root,a.root),c}function Ss(n,e){const t=n.root.querySelector("input"),i=t?Number.parseFloat(t.value):Number.NaN;return Number.isNaN(i)?e:i}function Fh(n){const e=E("div","panel-actions");return e.append(Ue({label:"Flip H",onClick:()=>n.flip("horizontal")}),Ue({label:"Flip V",onClick:()=>n.flip("vertical")}),Ue({label:"Duplicate",iconName:"copy",onClick:n.duplicateSelected}),Ue({label:"Delete",iconName:"trash",variant:"danger",onClick:n.deleteSelected})),e}const xm=[{mode:"left",label:"Left"},{mode:"center_horizontal",label:"Center"},{mode:"right",label:"Right"},{mode:"top",label:"Top"},{mode:"center_vertical",label:"Middle"},{mode:"bottom",label:"Bottom"}];function ym(n,e){const t=_n("Transform"),i=E("div","panel-note");i.textContent=`${n} workpieces selected`,t.body.appendChild(i);const s=E("div","num-field-label");s.textContent="Align";const r=E("div","align-grid");for(const{mode:o,label:a}of xm)r.appendChild(Ue({label:a,onClick:()=>e.align(o)}));return t.body.append(s,r),t.body.appendChild(Fh(e)),t.root}const Sm=["G54","G55","G56","G57","G58","G59","G59.1","G59.2","G59.3"];function bm(n,e){const t=_n("Layer settings"),i=bi({label:"Name",value:n.name,onCommit:d=>{d&&(En(n.uid,{name:d}),le.putLayer(n.uid,{name:d}).catch(f=>{_e("error",`Renaming failed: ${lr(f)}`)}))}}),s=E("div","schema-field-row layer-color-row"),r=E("span","num-field-label");r.textContent="Color";const o=E("input","layer-color-input");o.type="color",o.value=Mm(n.color),o.addEventListener("input",()=>{En(n.uid,{color:o.value}),le.putLayer(n.uid,{color:o.value}).catch(d=>{_e("error",`Color change failed: ${lr(d)}`)})}),s.append(r,o);const a=rs("WCS",[{value:"",label:"None"},...Sm.map(d=>({value:d,label:d}))],n.wcs??"",d=>{const f=d===""?null:d;En(n.uid,{wcs:f}),le.putLayer(n.uid,{wcs:f}).catch(g=>{_e("error",`WCS change failed: ${lr(g)}`)})});a.root.classList.add("in-panel");const l=E("div","schema-field-row"),c=E("span","num-field-label");c.textContent="Rotary";const u=os(!!n.rotary_enabled,d=>{En(n.uid,{rotary_enabled:d}),le.putLayer(n.uid,{rotary_enabled:d}).catch(f=>{_e("error",`Rotary toggle failed: ${lr(f)}`)})});l.append(c,u.root);const h=E("div","panel-actions layer-actions");return h.appendChild(Ue({label:"Add step",iconName:"zap",variant:"primary",onClick:()=>{Nh(n,e.refreshDoc)}})),t.body.append(i.root,s,a.root,l,h),t.root}function Mm(n){return/^#[0-9a-fA-F]{6}$/.test(n)?n:/^#[0-9a-fA-F]{3}$/.test(n)?`#${n[1]}${n[1]}${n[2]}${n[2]}${n[3]}${n[3]}`:"#4d9fff"}function wm(){const n=_n("Properties"),e=E("div","panel-note");return e.textContent="Nothing selected",n.body.appendChild(e),n.root}function Em(n){const e=E("div","command-palette"),t=E("input","command-input");t.type="text",t.spellcheck=!1,t.placeholder="Search commands",e.appendChild(Gt("search",13)),e.appendChild(t);const i=E("div","menu-popup command-popup");let s=[],r=0;function o(){i.remove()}function a(){return i.parentElement===e}function l(){const h=t.value.trim().toLowerCase();s=n.filter(f=>f.label.toLowerCase().includes(h)),r=0;const d=s.map((f,g)=>{const _=E("button","menu-item");_.type="button",g===r&&_.classList.add("active"),f.iconName&&_.appendChild(Gt(f.iconName,14));const m=E("span","menu-item-label");if(m.textContent=f.label,_.appendChild(m),f.accelerator){const p=E("span","menu-item-accel");p.textContent=f.accelerator,_.appendChild(p)}return _.addEventListener("click",()=>c(f)),_});if(d.length===0){const f=E("div","command-empty");f.textContent="No matching commands",i.replaceChildren(f);return}i.replaceChildren(...d)}function c(h){o(),t.value="",t.blur(),h.run()}function u(h){if(s.length===0)return;r=(r+h+s.length)%s.length;const d=Array.from(i.querySelectorAll(".menu-item"));d.forEach((f,g)=>f.classList.toggle("active",g===r)),d[r]?.scrollIntoView({block:"nearest"})}return t.addEventListener("focus",()=>{l(),e.appendChild(i)}),t.addEventListener("input",l),t.addEventListener("keydown",h=>{if(h.key==="ArrowDown")h.preventDefault(),u(1);else if(h.key==="ArrowUp")h.preventDefault(),u(-1);else if(h.key==="Enter"){h.preventDefault();const d=s[r];d&&c(d)}else h.key==="Escape"&&(o(),t.blur())}),document.addEventListener("pointerdown",h=>{a()&&!e.contains(h.target)&&o()}),{root:e,input:t}}function Tm(n){const e=E("div","topbar"),t=E("div","topbar-brand");t.append(Gt("box",16));const i=E("span");i.textContent="Rayforge",t.appendChild(i);const s=E("div","conn-pill disconnected"),r=E("span","conn-dot"),o=E("span","conn-text");o.textContent="Disconnected",s.append(r,o);const a=E("div","topbar-left");a.append(t,s),e.appendChild(a);const l=E("div","project-tab"),c=E("span","project-name");c.textContent=je.getState().projectName,l.appendChild(c),je.subscribe(p=>{c.textContent=p.projectName}),We.subscribe(p=>{l.title=p.dirty?"Unsaved changes":je.getState().projectName,l.classList.toggle("dirty",p.dirty)});const u=Ue({iconName:"undo",variant:"icon",title:"Undo (Ctrl+Z)",onClick:n.undo}),h=Ue({iconName:"redo",variant:"icon",title:"Redo (Ctrl+Y)",onClick:n.redo}),d=()=>{const{canUndo:p,canRedo:C}=We.getState();u.disabled=!p,h.disabled=!C};d(),We.subscribe(d);const f=[{label:"New project",accelerator:"Ctrl+N",iconName:"filePlus",run:n.newDoc},{label:"Open project",accelerator:"Ctrl+O",iconName:"folder",run:n.openProject},{label:"Save project",accelerator:"Ctrl+S",iconName:"save",run:n.saveProject},{label:"Export G-code",iconName:"save",run:n.exportGcode},{label:"Import file",accelerator:"Ctrl+I",iconName:"upload",run:n.importFile},{label:"Undo",accelerator:"Ctrl+Z",iconName:"undo",run:n.undo},{label:"Redo",accelerator:"Ctrl+Y",iconName:"redo",run:n.redo},{label:"Duplicate selection",accelerator:"Ctrl+D",iconName:"copy",run:n.duplicateSelected},{label:"Delete selection",accelerator:"Del",iconName:"trash",run:n.deleteSelected},{label:"Frame workarea",accelerator:"Home",iconName:"frame",run:n.frameSelection},{label:"Fit view",iconName:"fit",run:()=>n.viewPreset("fit")},{label:"Top view",accelerator:"1",iconName:"viewTop",run:()=>n.viewPreset("top")},{label:"Front view",accelerator:"2",iconName:"viewFront",run:()=>n.viewPreset("front")},{label:"Isometric view",accelerator:"3",iconName:"viewIso",run:()=>n.viewPreset("iso")},{label:"Toggle toolpaths",iconName:"eye",run:n.toggleToolpaths},{label:"Toggle travel moves",iconName:"eye",run:()=>Ch(!je.getState().showTravelMoves)},{label:"Toggle workpiece images",iconName:"eye",run:()=>Rh(!je.getState().showWorkpieceImages)},{label:"Toggle no-go zones",iconName:"eye",run:()=>Ph(!je.getState().showNogoZones)},{label:"Start job",iconName:"play",run:n.startJob},{label:"About Rayforge",iconName:"info",run:n.about}],g=Em(f),_=Ue({iconName:"sliders",variant:"icon",title:"App settings",onClick:n.openSettings}),m=E("div","topbar-actions");return m.append(u,h,_,g.root),e.append(l,m),{root:e,connectionPill:s,connectionText:o,projectTab:l,commandInput:g.input}}function Am(n){return n instanceof Error?n.message:String(n)}const Dl={phase:"idle",file:null,analysis:null,result:null,error:null};function Cm(n,e){const t=!e.supported||e.action==="unsupported"?"unsupported":e.action;return{...n,phase:t,analysis:e,error:null}}function Rm(n,e){return{...n,phase:"done",result:e,error:null}}function Il(n,e){return{...n,phase:"error",error:Am(e)}}class Pm{constructor(e){this.deps=e}deps;state=Dl;get current(){return this.state}async start(e){this.state={...Dl,phase:"analyzing",file:e};try{const t=await this.deps.analyze(e);this.state=Cm(this.state,t)}catch(t){this.state=Il(this.state,t)}return this.state}async run(e={}){const{file:t}=this.state;if(!t||this.state.phase==="importing")return this.state;this.state={...this.state,phase:"importing",error:null};try{const i=await this.deps.run(t,e);this.state=Rm(this.state,i)}catch(i){this.state=Il(this.state,i)}return this.state}}const Lm=128,Dm=96;function Im(n){return n.includes("BITMAP_TRACING")&&!n.includes("DIRECT_VECTOR")}function Nl(n){const e=E("input");e.type="file",e.style.display="none",e.addEventListener("change",()=>{const t=e.files?.[0];e.remove(),t&&Nm(t,n)}),document.body.appendChild(e),e.click()}async function Nm(n,e){const t=km(),i=t.querySelector(".dialog-body");i.appendChild(Fm(`Analyzing ${n.name}…`)),document.body.appendChild(t);const s=new Pm({analyze:o=>le.analyzeImport(o),run:(o,a)=>le.importFile(o,a)}),r=await s.start(n);if(r.phase==="unsupported"){ni(t),_e("error",`Unsupported file type: ${n.name}`);return}if(r.phase==="error"){ni(t),_e("error",`Import analysis failed: ${r.error}`);return}if(r.phase==="direct_load"){ni(t);const o=await s.run({});o.phase==="done"?await kh(o.result?.created.length??0,e,n.name):_e("error",`Import failed: ${o.error}`);return}if(r.phase!=="interactive_config"||!r.analysis){ni(t),_e("error","Import failed: unexpected analyzer result");return}Um(i,t,s,r.analysis,e,n.name)}async function kh(n,e,t){_e("success",n===1?`Imported ${t} (1 workpiece)`:`Imported ${t} (${n} workpieces)`),await e.refreshDoc()}function Um(n,e,t,i,s,r){Cn(n),n.appendChild(Om(i,r));const o=Im(i.features),a={threshold:Lm,invert:!1};let l=Dm;if(o){const p=E("div","import-spec-form"),C=Oh("Threshold",0,255,1,a.threshold,T=>{a.threshold=T}),M=os(!1,T=>{a.invert=T}),x=E("div","schema-field-row"),P=E("span","num-field-label");P.textContent="Invert",x.append(P,M.root);const w=un({label:"Pixels per inch",value:l,min:1,onCommit:T=>{l=T}});p.append(C.root,x,w.root),n.appendChild(p)}else{const p=E("div","panel-note");p.textContent="Vectors are imported as-is (passthrough).",n.appendChild(p)}let c=0,u=0;const h=E("div","field-grid import-placement"),d=un({label:"X (mm)",value:c,unit:"mm",onCommit:p=>{c=p}}),f=un({label:"Y (mm)",value:u,unit:"mm",onCommit:p=>{u=p}});h.append(d.root,f.root),n.appendChild(h);const g=E("div","dialog-footer"),_=Ue({label:"Cancel",onClick:()=>ni(e)}),m=Ue({label:"Import",variant:"primary",onClick:()=>{(async()=>{m.disabled=!0;const p=o?jp(a.threshold,a.invert,l):Zp(),C=await t.run({spec:p,x:c,y:u});C.phase==="done"?(ni(e),await kh(C.result?.created.length??0,s,r)):(m.disabled=!1,_e("error",`Import failed: ${C.error}`))})()}});g.append(_,m),n.appendChild(g)}function Om(n,e){const t=E("div","import-summary"),i=E("div","import-summary-title");i.textContent=e,t.appendChild(i);const s=E("div","panel-note"),r=[n.importer??"unknown importer"],o=n.manifest;if(o?.natural_size_mm){const[a,l]=o.natural_size_mm;r.push(`${a} × ${l} mm`)}o&&o.layers.length>0&&r.push(`${o.layers.length} layer(s)`),s.textContent=r.join(" · "),t.appendChild(s);for(const a of o?.warnings??[]){const l=E("div","panel-note import-warning");l.textContent=a,t.appendChild(l)}for(const a of o?.errors??[]){const l=E("div","panel-note import-warning");l.textContent=a,t.appendChild(l)}return t}function Fm(n){const e=E("div","panel-note");return e.textContent=n,e}function km(){const n=E("div","modal-backdrop"),e=E("div","dialog"),t=E("header","dialog-header"),i=E("span","dialog-title");i.textContent="Import";const s=Ue({iconName:"x",variant:"icon",title:"Close",onClick:()=>ni(n)});t.append(i,s);const r=E("div","dialog-body");return e.append(t,r),n.appendChild(e),n.addEventListener("click",o=>{o.target===n&&ni(n)}),n.dataset.role="import-dialog",n}function ni(n){n.remove()}function zm(n){return n instanceof Error?n.message:String(n)}function Bm(){const n=E("div","alarm-banner");n.classList.add("hidden"),n.appendChild(Gt("alert",16));const e=E("div","alarm-text"),t=E("span","alarm-title"),i=E("span","alarm-desc");e.append(t,i),n.appendChild(e);const s=Ue({label:"Clear alarm",variant:"danger",onClick:()=>{le.clearAlarm().then(()=>le.getMachineState()).then(r=>Ea(r)).catch(r=>_e("error",`Clear alarm failed: ${zm(r)}`))}});return n.appendChild(s),Et.subscribe(({machineState:r})=>{const o=r?.error??null,a=r?.status==="ALARM"||o!==null;n.classList.toggle("hidden",!a),a&&(t.textContent=o?.title||"Machine alarm",i.textContent=o?.description||`The machine reported state "${r?.status??"ALARM"}".`)}),n}const Hm=3e3;function Do(n){return n instanceof Error?n.message:String(n)}function Gm(){const n=E("div","job-bar"),e=Ue({label:"Start",iconName:"play",title:"Assemble and run the current document",onClick:()=>{le.startJob().catch(c=>_e("error",`Start failed: ${Do(c)}`))}}),t=Ue({label:"Pause",iconName:"pause",title:"Pause the running job",onClick:()=>{const c=Yt.getState().phase;le.holdJob(c!=="paused").catch(u=>_e("error",`Hold failed: ${Do(u)}`))}}),i=Ue({label:"Cancel",iconName:"stop",variant:"danger",title:"Cancel the running job (click again to confirm)",onClick:()=>{if(!i.classList.contains("armed")){i.classList.add("armed"),window.setTimeout(()=>i.classList.remove("armed"),Hm);return}i.classList.remove("armed"),le.cancelJob().catch(c=>_e("error",`Cancel failed: ${Do(c)}`))}}),s=E("div","job-progress"),r=E("div","job-progress-fill");s.appendChild(r);const o=E("span","job-phase");o.textContent="Idle";const a=E("span","job-eta mono");a.textContent="ETA --:--",n.append(e,t,i,s,o,a);const l=()=>{const{machineState:c}=Et.getState(),u=Yt.getState(),h=We.getState().layers.some(f=>f.workpieces.length>0),d=Lh({active:u.active,phase:u.phase,connected:c?.connected??!1,hasContent:h});e.disabled=!d.startEnabled,t.disabled=!d.holdEnabled,t.replaceChildren(Gt(d.holdIcon,14),(()=>{const f=E("span","btn-label");return f.textContent=d.holdLabel,f})()),i.disabled=!d.cancelEnabled,d.cancelEnabled||i.classList.remove("armed"),r.style.width=`${Math.round(Math.min(1,Math.max(0,u.progressFraction))*100)}%`,o.textContent=d.phaseLabel,a.textContent=u.active?`ETA ${vp(u.etaSeconds)}`:Vm()};return Et.subscribe(l),Yt.subscribe(l),We.subscribe(l),lt.subscribe(l),l(),n}function Vm(){const n=lt.getState();return n.estimatedSeconds!==null?`est ${Ca(n.estimatedSeconds)}`:"ETA --:--"}const Ul=200,Wm=14;function Yr(n,e){const t=Math.max(0,n);return{undos:Math.max(0,e-t),redos:Math.max(0,t-e)}}function Ol(n,e,t){const i=n.length+e.length;return t<0||t>=i?null:Yr(t+1,n.length)}function Fl(n,e=Wm){return n.length<=e?n:`${n.slice(0,e-1)}…`}function kl(n,e){let t=0,i=1/0;for(let s=0;s<e.length;s++){const r=Math.abs(e[s]-n);r<i&&(i=r,t=s)}return t}function Xm(n){const e=E("div","history-timeline"),t=E("span","history-label");t.textContent="History";const i=Ue({iconName:"stepBack",variant:"icon",title:"Jump to start (undo everything)",onClick:()=>{p(Yr(0,h()))}}),s=Ue({iconName:"chevronLeft",variant:"icon",title:"Step back (undo)",onClick:()=>{p({undos:1,redos:0})}}),r=Ue({iconName:"chevronRight",variant:"icon",title:"Step forward (redo)",onClick:()=>{p({undos:0,redos:1})}}),o=Ue({iconName:"stepForward",variant:"icon",title:"Jump to end (redo everything)",onClick:()=>{p(Yr(d(),h()))}}),a=E("div","history-chips"),l=E("div","history-handle");l.title="Drag to roll the document back or forward";let c=!1,u=[];const h=()=>We.getState().undoLabels.length,d=()=>{const M=We.getState();return M.undoLabels.length+M.redoLabels.length};a.appendChild(l);function f(){const M=We.getState(),x=[];M.undoLabels.forEach((P,w)=>{const T=E("button","history-chip");T.type="button",T.textContent=Fl(P),T.title=P,T.addEventListener("click",()=>{const L=Ol(M.undoLabels,M.redoLabels,w);L&&p(L)}),x.push(T)}),M.redoLabels.forEach((P,w)=>{const T=E("button","history-chip redo");T.type="button",T.textContent=Fl(P),T.title=`Redo: ${P}`,T.addEventListener("click",()=>{const L=Ol(M.undoLabels,M.redoLabels,M.undoLabels.length+w);L&&p(L)}),x.push(T)});for(const P of[...a.children])P!==l&&P.remove();a.append(...x),_()}function g(){const M=Array.from(a.querySelectorAll(".history-chip")),x=[0];for(const P of M)x.push(P.offsetLeft+P.offsetWidth);return x}function _(){const M=We.getState();u=g();const x=M.undoLabels.length,P=x===0?u[0]:u[Math.min(x,u.length-1)];l.style.left=`${P}px`,l.classList.toggle("dragging",!1)}function m(){const M=We.getState(),x=c;i.disabled=x||!M.canUndo,s.disabled=x||!M.canUndo,r.disabled=x||!M.canRedo,o.disabled=x||!M.canRedo;for(const P of Array.from(a.querySelectorAll(".history-chip")))P.disabled=x;e.classList.toggle("rolling",x)}async function p(M){if(c)return;const x=Math.min(M.undos,Ul),P=Math.min(M.redos,Ul);if(!(x===0&&P===0)){c=!0,gl(!0),m();try{for(let w=0;w<x;w++)await n.api.postUndo();for(let w=0;w<P;w++)await n.api.postRedo()}catch(w){n.onError(`History roll failed: ${w instanceof Error?w.message:String(w)}`)}finally{c=!1,gl(!1),m(),n.onRollEnd()}}}l.addEventListener("pointerdown",M=>{M.preventDefault(),l.setPointerCapture(M.pointerId),l.classList.add("dragging")}),l.addEventListener("pointermove",M=>{if(!l.classList.contains("dragging"))return;const x=a.getBoundingClientRect(),P=kl(M.clientX-x.left,u);l.style.left=`${u[P]??0}px`});const C=M=>{if(!l.classList.contains("dragging"))return;l.classList.remove("dragging");const x=a.getBoundingClientRect(),P=kl(M.clientX-x.left,u),w=We.getState(),T=Yr(P,w.undoLabels.length);p(T)};return l.addEventListener("pointerup",C),l.addEventListener("pointercancel",C),e.append(t,i,s,r,o,a),We.subscribe(()=>{f(),m()}),m(),{root:e}}function $m(n){return n.length===0?{kind:"none"}:n.length===1?{kind:"run",layout:n[0]}:{kind:"picker",layouts:n}}function Ym(n,e,t){const i=E("div","menu-popup layout-picker");for(const l of e){const c=E("button","menu-item");c.type="button";const u=E("span","menu-item-label");u.textContent=l.label,c.appendChild(u),c.addEventListener("click",()=>{s(),t(l)}),i.appendChild(c)}function s(){i.remove(),document.removeEventListener("pointerdown",r,!0),document.removeEventListener("keydown",o,!0)}function r(l){i.contains(l.target)||(s(),t(null))}function o(l){l.key==="Escape"&&(s(),t(null))}document.addEventListener("pointerdown",r,!0),document.addEventListener("keydown",o,!0);const a=n.getBoundingClientRect();return i.style.left=`${a.left}px`,i.style.top=`${a.bottom+6}px`,document.body.appendChild(i),{root:i,close:s}}function qm(n,e){return new Promise(t=>{Ym(n,e,t)})}function zh(n,e){let t=n;for(const i of e.split(".")){if(t==null)return;t=t[i]}return t}function Ra(n,e,t){const[i,...s]=e.split(".");if(s.length===0)return{...n,[i]:t};const r=n[i];if(Array.isArray(r)){const a=[...r],l=Number(s[0]);if(s.length===1)a[l]=t;else{const c=r[l]??{};a[l]=Ra(c,s.join("."),t)}return{...n,[i]:a}}const o=r??{};return{...n,[i]:Ra(o,s.join("."),t)}}function Pa(n,e){const t=[],i=E("div","settings-section");for(const s of n){const{element:r,read:o}=Zm(s,e);t.push({key:s.key,initial:zh(e,s.key),read:o}),i.appendChild(r)}return{root:i,collect(){const s={};for(const r of t){const o=r.read();o!==void 0&&o!==r.initial&&String(o)!==String(r.initial)&&(s[r.key]=o)}return s}}}function zl(n,e){const t={...n};for(const[s,r]of Object.entries(e)){const o=Ra(t,s,r),a=s.split(".")[0];t[a]=o[a]}const i={};for(const s of Object.keys(e)){const r=s.split(".")[0];i[r]=t[r]}return i}function Bl(n){return n instanceof Error?n.message:String(n)}async function jm(n){let e;try{e=await n.load()}catch(m){_e("error",`Loading settings failed: ${Bl(m)}`);return}const t=n.sections.map(m=>Pa(m.fields,e)),i=E("div","dialog-body settings-body"),s=new Map,r=n.sections.map((m,p)=>({id:String(p),label:m.label})),o=_o(r,r[0]?.id??"0",m=>{for(const[p,C]of s)C.hidden=p!==m});i.appendChild(o.root),n.sections.forEach((m,p)=>{const C=t[p].root;if(p>0&&(C.hidden=!0),m.note){const M=E("div","panel-note");M.textContent=m.note,C.appendChild(M)}s.set(String(p),C),i.appendChild(C)});const a=E("div","panel-note settings-error"),l=Ue({label:"Save",iconName:"save",variant:"primary",onClick:()=>{g()}}),c=E("div","panel-actions");c.appendChild(l),i.appendChild(c);const u=E("div","modal-backdrop"),h=E("div","dialog"),d=E("header","dialog-header"),f=E("span","dialog-title");f.textContent=n.title,d.append(f,Ue({iconName:"x",variant:"icon",title:"Close",onClick:_})),h.append(d,i),u.appendChild(h),u.addEventListener("click",m=>{m.target===u&&_()}),document.body.appendChild(u);async function g(){const m={};for(const p of t)Object.assign(m,p.collect());if(Object.keys(m).length===0){_();return}l.disabled=!0;try{await n.save(m),_(),_e("success",n.savedMessage),n.onSaved?.()}catch(p){a.textContent=Bl(p),a.parentElement||i.insertBefore(a,c),l.disabled=!1}}function _(){u.remove()}}function Zm(n,e){const t=zh(e,n.key);if(n.kind==="switch"){const s=os(!!t,()=>{}),r=E("div","schema-field-row"),o=E("span","num-field-label");return o.textContent=n.label,r.append(o,s.root),{element:r,read:()=>s.root.classList.contains("on")}}if(n.kind==="choice"){const s=n.options??[],r=String(t??s[0]?.value??""),o=rs(n.label,s,r,()=>{});return o.root.classList.add("in-panel"),{element:o.root,read:()=>{const a=o.root.querySelector("select");return a?a.value:void 0}}}if(n.kind==="number"){const s=un({label:n.label,value:typeof t=="number"?t:0,unit:n.unit,onCommit:()=>{}});return{element:s.root,read:()=>{const r=s.root.querySelector("input");if(!r)return;const o=Number.parseFloat(r.value);return Number.isNaN(o)?void 0:o}}}const i=bi({label:n.label,value:String(t??""),onCommit:()=>{}});return{element:i.root,read:()=>{const s=i.root.querySelector("input");if(!s)return;const r=s.value.trim();return r===""?void 0:r}}}const Jm=[{value:"bottom_left",label:"Bottom left"},{value:"bottom_right",label:"Bottom right"},{value:"top_left",label:"Top left"},{value:"top_right",label:"Top right"}],Km=[{key:"name",label:"Name",kind:"text"},{key:"origin",label:"Origin corner",kind:"choice",options:Jm},{key:"axis_extents.0",label:"Bed width",kind:"number",unit:"mm"},{key:"axis_extents.1",label:"Bed height",kind:"number",unit:"mm"},{key:"work_margins.0",label:"Margin left",kind:"number",unit:"mm"},{key:"work_margins.1",label:"Margin top",kind:"number",unit:"mm"},{key:"work_margins.2",label:"Margin right",kind:"number",unit:"mm"},{key:"work_margins.3",label:"Margin bottom",kind:"number",unit:"mm"},{key:"reverse_x_axis",label:"Reverse X",kind:"switch"},{key:"reverse_y_axis",label:"Reverse Y",kind:"switch"},{key:"reverse_z_axis",label:"Reverse Z",kind:"switch"},{key:"soft_limits_enabled",label:"Soft limits",kind:"switch"},{key:"max_travel_speed",label:"Max travel speed",kind:"number",unit:"mm/min"},{key:"max_cut_speed",label:"Max cut speed",kind:"number",unit:"mm/min"},{key:"acceleration",label:"Acceleration",kind:"number",unit:"mm/s²"},{key:"jog_speed",label:"Jog speed",kind:"number",unit:"mm/min"}],Qm=[{key:"max_power",label:"Max power (0-1000)",kind:"number"},{key:"frame_power",label:"Frame power (0-1)",kind:"number"},{key:"focus_power",label:"Focus power (0-1)",kind:"number"},{key:"frame_speed",label:"Frame speed (0 = machine default)",kind:"number",unit:"mm/min"},{key:"frame_repeat_count",label:"Frame repeat count",kind:"number"},{key:"frame_corner_pause",label:"Frame corner pause (s)",kind:"number"},{key:"spot_size_mm.0",label:"Spot width",kind:"number",unit:"mm"},{key:"spot_size_mm.1",label:"Spot height",kind:"number",unit:"mm"},{key:"focal_distance",label:"Focal distance (0 = auto)",kind:"number",unit:"mm"},{key:"pointer_offset_enabled",label:"Pointer offset",kind:"switch"},{key:"pointer_offset_x",label:"Pointer offset X",kind:"number",unit:"mm"},{key:"pointer_offset_y",label:"Pointer offset Y",kind:"number",unit:"mm"}];function Sn(n){return n instanceof Error?n.message:String(n)}async function eg(n){let e;try{const[w,T,L,y]=await Promise.all([le.getMachineSettings(n.machineId),le.getDriverSetup(n.machineId),le.getDrivers().catch(()=>[]),le.getMacros(n.machineId).catch(()=>[])]);e={settings:w,driverSetup:T,drivers:L,macros:y}}catch(w){_e("error",`Loading machine settings failed: ${Sn(w)}`);return}const t=E("div","dialog-body settings-body machine-settings"),i=E("div","machine-settings-content");t.appendChild(i);const s=E("div","modal-backdrop"),r=E("div","dialog dialog-wide"),o=E("header","dialog-header"),a=E("span","dialog-title");a.textContent=`Machine settings — ${e.settings.name}`,o.append(a,Ue({iconName:"x",variant:"icon",title:"Close",onClick:l})),r.append(o,t),s.appendChild(r),s.addEventListener("click",w=>{w.target===s&&l()}),document.body.appendChild(s);function l(){s.remove()}function c(w){for(const[T,L]of u)L.hidden=T!==w}const u=new Map,d=_o([{id:"general",label:"General"},{id:"device",label:"Device"},{id:"laser",label:"Laser"},{id:"macros",label:"Macros"},{id:"offsets",label:"Work offsets"},{id:"nogo",label:"No-go zones"},{id:"rotary",label:"Rotary"}],"general",w=>{d.setActive(w),c(w)});t.insertBefore(d.root,i),u.set("general",g()),u.set("device",m(e.driverSetup,e.drivers)),u.set("laser",p(e.settings)),u.set("macros",C()),u.set("offsets",M(e.settings.coordinate_systems)),u.set("nogo",x(e.settings.nogo_zones)),u.set("rotary",P(e.settings));for(const[,w]of u)i.appendChild(w);c("general");function f(w){const T=E("div","panel-note settings-error");return w.appendChild(T),T}function g(){const w=E("div","settings-section"),T=Pa(Km,e.settings);return w.appendChild(T.root),w.appendChild(_(w,async()=>{const L=T.collect();Object.keys(L).length!==0&&await le.putMachineSettings(n.machineId,zl(e.settings,L))})),w}function _(w,T){const L=E("div","panel-actions"),y=f(w),b=Ue({label:"Save",iconName:"save",variant:"primary",onClick:()=>{b.disabled=!0,T().then(()=>{l(),_e("success","Machine settings saved"),n.onSaved?.()}).catch(N=>{y.textContent=Sn(N),b.disabled=!1})}});return L.appendChild(b),L}function m(w,T){const L=E("div","settings-section"),y={...w.values},b=new Set;let N=w.driver;const O=E("div","schema-field-row"),G=E("span","num-field-label");G.textContent="Driver";const V=rs("Driver",T.map(Q=>({value:Q.name,label:Q.name.replace(/Driver$/,"")+(Q.maturity_warning?` (${Q.maturity})`:"")})),w.driver,Q=>{N=Q});V.root.classList.add("in-panel"),O.append(G,V.root),L.appendChild(O);const X=E("div","panel-note");if(X.textContent="Switching the driver resets its settings to the defaults.",L.appendChild(X),w.precheck_error){const Q=E("div","panel-note settings-error");Q.textContent=`Precheck: ${w.precheck_error}`,L.appendChild(Q)}const q=Rc({groups:[w.setup]},w.values),J=E("div","schema-group"),z=E("div","schema-group-title");z.textContent="Connection settings",J.appendChild(z);for(const Q of q)for(const me of Q.fields)J.appendChild(Lc(me,(De,Ie)=>{y[De]=Ie,b.add(De)}));if(L.appendChild(J),w.device_settings&&w.device_settings.length>0){const Q=E("div","panel-note");Q.textContent="Device ($$) settings are read from the connected hardware; editing them from the web UI is not available yet.",L.appendChild(Q),tg(L,w.device_settings,w.values)}const H=E("div","panel-actions"),D=E("div","panel-note settings-error"),W=Ue({label:"Save",iconName:"save",variant:"primary",onClick:()=>{W.disabled=!0,(async()=>{try{const Q={};for(const me of b)Q[me]=y[me];N!==w.driver&&await le.putDriver(n.machineId,N),Object.keys(Q).length>0&&await le.putDriverSetup(n.machineId,Q),l(),_e("success","Device settings saved"),n.onSaved?.()}catch(Q){D.textContent=Sn(Q),W.disabled=!1}})()}});return H.appendChild(W),L.appendChild(H),L}function p(w){const T=E("div","settings-section"),L=[];for(const y of w.heads){const b=E("div","schema-group"),N=E("div","schema-group-title");N.textContent=y.name||y.type,b.appendChild(N);const O={...y};delete O.uid,delete O.name,delete O.type;const G=Pa(Qm,O);L.push({uid:y.uid,collect:()=>zl(O,G.collect())}),b.appendChild(G.root),T.appendChild(b)}if(w.heads.length===0){const y=E("div","panel-note");y.textContent="This machine has no laser heads configured.",T.appendChild(y)}return T.appendChild(_(T,async()=>{await le.putHeads(n.machineId,L.map(y=>({uid:y.uid,...y.collect()})))})),T}function C(){const w=E("div","settings-section");let T=[...e.macros];const L=E("div","macro-list");w.appendChild(L);const y=E("div","panel-note settings-error"),b=()=>{for(const V of[...L.children])V.remove();if(T.length===0){const V=E("div","panel-note");V.textContent="No macros defined.",L.appendChild(V)}for(const V of T)L.appendChild(G(V))},N=Ue({label:"Add macro",iconName:"filePlus",onClick:()=>{le.createMacro(n.machineId,{name:`Macro ${T.length+1}`,code:"",enabled:!0}).then(V=>{T.push(V),b()}).catch(V=>_e("error",`Adding macro failed: ${Sn(V)}`))}}),O=E("div","panel-actions");O.append(N,y),w.appendChild(O);function G(V){const X=E("div","schema-group macro-item"),q=bi({label:"Name",value:V.name,onCommit:()=>{}}),J=E("span","num-field-label");J.textContent="G-code";const z=E("textarea","textarea-input");z.value=V.code.join(`
`),z.spellcheck=!1,z.rows=4;const H=E("div","schema-field-row"),D=E("span","num-field-label");D.textContent="Enabled";const W=os(V.enabled,()=>{});H.append(D,W.root);const Q=E("div","panel-actions"),me=E("div","panel-note settings-error"),De=Ue({label:"Save",onClick:()=>{De.disabled=!0,le.updateMacro(n.machineId,V.uid,{name:ur(q),code:z.value,enabled:W.root.classList.contains("on")}).then(j=>{Object.assign(V,j),_e("success","Macro saved")}).catch(j=>{me.textContent=Sn(j),De.disabled=!1})}}),Ie=Ue({label:"Delete",variant:"danger",onClick:()=>{le.deleteMacro(n.machineId,V.uid).then(()=>{T=T.filter(j=>j.uid!==V.uid),X.remove()}).catch(j=>{me.textContent=Sn(j)})}});return Q.append(De,Ie,me),X.append(q.root,J,z,H,Q),X}return b(),w}function M(w){const T=E("div","settings-section"),L=E("div","panel-note");L.textContent="Work offsets are sent to the machine when a job starts; use Sync in the control panel to re-read them from the device.",T.appendChild(L);for(const y of w){const b=E("div","schema-group wcs-row"),N=E("div","schema-group-title");N.textContent=y.name,b.appendChild(N);const O=bi({label:"Label",value:y.label,onCommit:()=>{}}),G=["X","Y","Z"].map((z,H)=>({axis:z,handle:un({label:z,value:y.offset[H]??0,onCommit:()=>{}})})),V=E("div","field-grid");for(const z of G)V.appendChild(z.handle.root);b.append(O.root,V);const X=E("div","panel-actions"),q=E("div","panel-note settings-error"),J=Ue({label:"Save",onClick:()=>{J.disabled=!0,le.putWcs(n.machineId,y.name,G.map(z=>hr(z.handle,0)),ur(O)).then(()=>{_e("success",`${y.name} saved`)}).catch(z=>{q.textContent=Sn(z),J.disabled=!1})}});X.append(J,q),b.appendChild(X),T.appendChild(b)}return T}function x(w){const T=E("div","settings-section"),L=E("div","nogo-list"),y=()=>{for(const H of[...L.children])H.remove();if(w.length===0){const H=E("div","panel-note");H.textContent="No no-go zones defined.",L.appendChild(H)}for(const H of w)L.appendChild(b(H))},b=H=>{const D=E("div","schema-group nogo-item"),W=E("div","schema-group-title");W.textContent=H.name||"No-Go Zone",D.appendChild(W);const Q=E("div","panel-note");Q.textContent=`${H.shape} · `+Object.entries(H.params).map(([Ie,j])=>`${Ie}=${j}`).join(", "),D.appendChild(Q);const me=E("div","panel-actions"),De=E("div","panel-note settings-error");return me.append(Ue({label:"Delete",variant:"danger",onClick:()=>{le.deleteNogoZone(n.machineId,H.uid).then(()=>{const Ie=w.indexOf(H);Ie>=0&&w.splice(Ie,1),D.remove(),n.onSaved?.()}).catch(Ie=>{De.textContent=Sn(Ie)})}}),De),D.appendChild(me),D};y(),T.appendChild(L);const N=E("div","schema-group-title");N.textContent="Add zone",T.appendChild(N);const O=bi({label:"Name",value:"No-Go Zone",onCommit:()=>{}}),G=rs("Shape",[{value:"rect",label:"Rectangle"},{value:"cylinder",label:"Cylinder"}],"rect",()=>{}),V=new Map,X=[["x","X"],["y","Y"],["w","Width"],["h","Height"],["radius","Radius"]];for(const[H,D]of X)V.set(H,un({label:D,value:H==="w"||H==="h"?20:10}));const q=E("div","field-grid");for(const[,H]of V)q.appendChild(H.root);T.append(O.root,G.root,q);const J=E("div","panel-actions"),z=E("div","panel-note settings-error");return J.append(Ue({label:"Add zone",iconName:"filePlus",onClick:()=>{const H=ng(G),D={},W=H==="cylinder"?["x","y","radius"]:["x","y","w","h"];for(const Q of W){const me=V.get(Q);me&&(D[Q]=hr(me,0))}le.createNogoZone(n.machineId,{name:ur(O),shape:H,params:D}).then(Q=>{w.push(Q),y(),n.onSaved?.()}).catch(Q=>{z.textContent=Sn(Q)})}}),z),T.appendChild(J),T}function P(w){const T=E("div","settings-section");if(w.rotary_modules.length===0){const L=E("div","panel-note");L.textContent="No rotary modules configured. Rotary modules are created in the desktop app; their diameter and feed settings can be edited here once they exist.",T.appendChild(L)}for(const L of w.rotary_modules){const y=E("div","schema-group"),b=E("div","schema-group-title");b.textContent=`${L.name} (${L.axis})`,y.appendChild(b);const N=bi({label:"Name",value:L.name,onCommit:()=>{}}),O=un({label:"Default diameter",value:L.default_diameter,unit:"mm",onCommit:()=>{}}),G=un({label:"Movement per rotation",value:L.mm_per_rotation,unit:"mm",onCommit:()=>{}}),V=E("div","schema-field-row"),X=E("span","num-field-label");X.textContent="Reverse";const q=os(L.reverse_axis,()=>{});V.append(X,q.root);const J=E("div","panel-actions"),z=E("div","panel-note settings-error"),H=Ue({label:"Save",onClick:()=>{H.disabled=!0,le.putRotaryModule(n.machineId,L.uid,{name:ur(N),default_diameter:hr(O,L.default_diameter),mm_per_rotation:hr(G,L.mm_per_rotation),reverse_axis:q.root.classList.contains("on")}).then(()=>_e("success","Rotary module saved")).catch(D=>{z.textContent=Sn(D),H.disabled=!1})}});J.append(H,z),y.append(N.root,O.root,G.root,V,J),T.appendChild(y)}return T}}function tg(n,e,t){for(const i of Rc({groups:e},t)){const s=E("div","schema-group"),r=E("div","schema-group-title");r.textContent=i.title||"Device settings",s.appendChild(r);for(const o of i.fields){const a=Lc(o,()=>{});a.querySelectorAll("input, textarea, select, button").forEach(l=>{l.disabled=!0}),s.appendChild(a)}n.appendChild(s)}}function ur(n){const e=n.root.querySelector("input");return e?e.value:""}function ng(n){const e=n.root.querySelector("select");return e?e.value:""}function hr(n,e){const t=n.root.querySelector("input"),i=t?Number.parseFloat(t.value):Number.NaN;return Number.isNaN(i)?e:i}function ig(){const n=E("footer","statusbar"),e=E("button","status-chip");e.type="button",e.title="Open machine tab";const t=E("span","status-dot off"),i=E("span","status-text");i.textContent="Connecting…";const s=E("span","status-item mono"),r=E("span","status-item");e.append(t,i,s,r),e.addEventListener("click",()=>Th("machine")),n.append(e);const o=()=>{const{machineState:a,viewport:l}=Et.getState(),c=a?.connected??!1;t.className=`status-dot ${c?"on":"off"}`,i.textContent=a?`${c?"Connected":"Offline"} · ${a.status}`:"Connecting…";const u=a?.beam_position_panel??l?.beam_position_panel??null;s.textContent=u?`beam ${Ml(u.x)}, ${Ml(u.y)} mm`:"";const h=l?.active_wcs;r.textContent=l?`${h?.name??"G54"} · ${h?.label??""} · rotary ${l.rotary_active?"on":"off"}`:""};return Et.subscribe(o),o(),{root:n,statusChip:e,statusText:i,beamPos:s,wcsText:r}}function sg(){const n=E("div","transport-pill"),e=Ue({iconName:"stepBack",variant:"icon",title:"Step to previous segment",disabled:!0,onClick:()=>rp()}),t=Ue({iconName:"play",variant:"icon",title:"Play toolpath preview",disabled:!0,onClick:()=>ep()}),i=Ue({iconName:"pause",variant:"icon",title:"Pause preview",disabled:!0,onClick:()=>tp()});i.hidden=!0;const s=Ue({iconName:"stepForward",variant:"icon",title:"Step to next segment",disabled:!0,onClick:()=>sp()}),r=E("input","transport-scrubber");r.type="range",r.min="0",r.max="0",r.step="0.05",r.value="0",r.disabled=!0,r.addEventListener("input",()=>{np(Number.parseFloat(r.value))}),r.addEventListener("pointerdown",()=>rr(!0)),r.addEventListener("pointerup",()=>rr(!1)),r.addEventListener("focus",()=>rr(!0)),r.addEventListener("blur",()=>rr(!1));const o=E("span","transport-time mono");o.textContent="– / –";const a=E("select","transport-speed");for(const c of Zf){const u=E("option");u.value=String(c),u.textContent=`${c}x`,a.appendChild(u)}a.disabled=!0,a.addEventListener("change",()=>{ip(Number.parseFloat(a.value))}),n.append(e,t,i,s,r,o,a);const l=()=>{const c=lt.getState(),u=Yt.getState(),h=c.track,d=u.active,f=h!==null&&!d;if(e.disabled=!f,s.disabled=!f,t.disabled=!f,i.disabled=!f,r.disabled=!f,a.disabled=!f,t.hidden=c.playing,i.hidden=!c.playing,h){r.max=String(h.totalDistance);const g=d?u.progressFraction:h.totalDistance>0?c.positionMm/h.totalDistance:0;r.value=String(d?g*h.totalDistance:c.positionMm);const _=c.estimatedSeconds;o.textContent=`${Ca(_!==null?g*_:null)} / ${Ca(_)}`}else r.max="0",r.value="0",o.textContent="– / –"};return lt.subscribe(l),Yt.subscribe(l),l(),{root:n,scrubber:r}}const rg=5e3,og={info:"info",success:"check",warning:"alert",error:"x"};function ag(){const n=E("div","toasts");let e=new Set;return je.subscribe(t=>{const i=t.toasts;for(const s of i)e.has(s.id)||window.setTimeout(()=>Eh(s.id),rg);e=new Set(i.map(s=>s.id)),cg(n,i)}),n}function cg(n,e){Cn(n);for(const t of e){const i=E("div",`toast toast-${t.level}`);i.appendChild(Gt(og[t.level]??"info",14));const s=E("span","toast-msg");s.textContent=t.message;const r=E("button","toast-close");r.type="button",r.appendChild(Gt("x",12)),r.addEventListener("click",()=>Eh(t.id)),i.append(s,r),n.appendChild(i)}}const Ic="180",ts={ROTATE:0,DOLLY:1,PAN:2},Qi={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},lg=0,Hl=1,ug=2,Bh=1,hg=2,Bn=3,oi=0,Jt=1,cn=2,si=0,ns=1,Gl=2,Vl=3,Wl=4,dg=5,yi=100,fg=101,pg=102,mg=103,gg=104,_g=200,vg=201,xg=202,yg=203,La=204,Da=205,Sg=206,bg=207,Mg=208,wg=209,Eg=210,Tg=211,Ag=212,Cg=213,Rg=214,Ia=0,Na=1,Ua=2,as=3,Oa=4,Fa=5,ka=6,za=7,Hh=0,Pg=1,Lg=2,ri=0,Dg=1,Ig=2,Ng=3,Ug=4,Og=5,Fg=6,kg=7,Gh=300,cs=301,ls=302,Ba=303,Ha=304,vo=306,Ga=1e3,Mi=1001,Va=1002,xn=1003,zg=1004,dr=1005,Tn=1006,Io=1007,wi=1008,Rn=1009,Vh=1010,Wh=1011,Bs=1012,Nc=1013,Ei=1014,Wn=1015,Ks=1016,Uc=1017,Oc=1018,Hs=1020,Xh=35902,$h=35899,Yh=1021,qh=1022,vn=1023,Gs=1026,Vs=1027,jh=1028,Fc=1029,Zh=1030,kc=1031,zc=1033,qr=33776,jr=33777,Zr=33778,Jr=33779,Wa=35840,Xa=35841,$a=35842,Ya=35843,qa=36196,ja=37492,Za=37496,Ja=37808,Ka=37809,Qa=37810,ec=37811,tc=37812,nc=37813,ic=37814,sc=37815,rc=37816,oc=37817,ac=37818,cc=37819,lc=37820,uc=37821,hc=36492,dc=36494,fc=36495,pc=36283,mc=36284,gc=36285,_c=36286,Bg=3200,Hg=3201,Gg=0,Vg=1,ti="",Ot="srgb",us="srgb-linear",oo="linear",rt="srgb",Di=7680,Xl=519,Wg=512,Xg=513,$g=514,Jh=515,Yg=516,qg=517,jg=518,Zg=519,vc=35044,$l="300 es",An=2e3,ao=2001;class Ri{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const i=this._listeners;i[e]===void 0&&(i[e]=[]),i[e].indexOf(t)===-1&&i[e].push(t)}hasEventListener(e,t){const i=this._listeners;return i===void 0?!1:i[e]!==void 0&&i[e].indexOf(t)!==-1}removeEventListener(e,t){const i=this._listeners;if(i===void 0)return;const s=i[e];if(s!==void 0){const r=s.indexOf(t);r!==-1&&s.splice(r,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const i=t[e.type];if(i!==void 0){e.target=this;const s=i.slice(0);for(let r=0,o=s.length;r<o;r++)s[r].call(this,e);e.target=null}}}const zt=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],Kr=Math.PI/180,xc=180/Math.PI;function Xn(){const n=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,i=Math.random()*4294967295|0;return(zt[n&255]+zt[n>>8&255]+zt[n>>16&255]+zt[n>>24&255]+"-"+zt[e&255]+zt[e>>8&255]+"-"+zt[e>>16&15|64]+zt[e>>24&255]+"-"+zt[t&63|128]+zt[t>>8&255]+"-"+zt[t>>16&255]+zt[t>>24&255]+zt[i&255]+zt[i>>8&255]+zt[i>>16&255]+zt[i>>24&255]).toLowerCase()}function Ze(n,e,t){return Math.max(e,Math.min(t,n))}function Jg(n,e){return(n%e+e)%e}function No(n,e,t){return(1-t)*n+t*e}function wn(n,e){switch(e.constructor){case Float32Array:return n;case Uint32Array:return n/4294967295;case Uint16Array:return n/65535;case Uint8Array:return n/255;case Int32Array:return Math.max(n/2147483647,-1);case Int16Array:return Math.max(n/32767,-1);case Int8Array:return Math.max(n/127,-1);default:throw new Error("Invalid component type.")}}function ot(n,e){switch(e.constructor){case Float32Array:return n;case Uint32Array:return Math.round(n*4294967295);case Uint16Array:return Math.round(n*65535);case Uint8Array:return Math.round(n*255);case Int32Array:return Math.round(n*2147483647);case Int16Array:return Math.round(n*32767);case Int8Array:return Math.round(n*127);default:throw new Error("Invalid component type.")}}const Kg={DEG2RAD:Kr};class ue{constructor(e=0,t=0){ue.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,i=this.y,s=e.elements;return this.x=s[0]*t+s[3]*i+s[6],this.y=s[1]*t+s[4]*i+s[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=Ze(this.x,e.x,t.x),this.y=Ze(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=Ze(this.x,e,t),this.y=Ze(this.y,e,t),this}clampLength(e,t){const i=this.length();return this.divideScalar(i||1).multiplyScalar(Ze(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const i=this.dot(e)/t;return Math.acos(Ze(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,i=this.y-e.y;return t*t+i*i}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const i=Math.cos(t),s=Math.sin(t),r=this.x-e.x,o=this.y-e.y;return this.x=r*i-o*s+e.x,this.y=r*s+o*i+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class Ti{constructor(e=0,t=0,i=0,s=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=i,this._w=s}static slerpFlat(e,t,i,s,r,o,a){let l=i[s+0],c=i[s+1],u=i[s+2],h=i[s+3];const d=r[o+0],f=r[o+1],g=r[o+2],_=r[o+3];if(a===0){e[t+0]=l,e[t+1]=c,e[t+2]=u,e[t+3]=h;return}if(a===1){e[t+0]=d,e[t+1]=f,e[t+2]=g,e[t+3]=_;return}if(h!==_||l!==d||c!==f||u!==g){let m=1-a;const p=l*d+c*f+u*g+h*_,C=p>=0?1:-1,M=1-p*p;if(M>Number.EPSILON){const P=Math.sqrt(M),w=Math.atan2(P,p*C);m=Math.sin(m*w)/P,a=Math.sin(a*w)/P}const x=a*C;if(l=l*m+d*x,c=c*m+f*x,u=u*m+g*x,h=h*m+_*x,m===1-a){const P=1/Math.sqrt(l*l+c*c+u*u+h*h);l*=P,c*=P,u*=P,h*=P}}e[t]=l,e[t+1]=c,e[t+2]=u,e[t+3]=h}static multiplyQuaternionsFlat(e,t,i,s,r,o){const a=i[s],l=i[s+1],c=i[s+2],u=i[s+3],h=r[o],d=r[o+1],f=r[o+2],g=r[o+3];return e[t]=a*g+u*h+l*f-c*d,e[t+1]=l*g+u*d+c*h-a*f,e[t+2]=c*g+u*f+a*d-l*h,e[t+3]=u*g-a*h-l*d-c*f,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,i,s){return this._x=e,this._y=t,this._z=i,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const i=e._x,s=e._y,r=e._z,o=e._order,a=Math.cos,l=Math.sin,c=a(i/2),u=a(s/2),h=a(r/2),d=l(i/2),f=l(s/2),g=l(r/2);switch(o){case"XYZ":this._x=d*u*h+c*f*g,this._y=c*f*h-d*u*g,this._z=c*u*g+d*f*h,this._w=c*u*h-d*f*g;break;case"YXZ":this._x=d*u*h+c*f*g,this._y=c*f*h-d*u*g,this._z=c*u*g-d*f*h,this._w=c*u*h+d*f*g;break;case"ZXY":this._x=d*u*h-c*f*g,this._y=c*f*h+d*u*g,this._z=c*u*g+d*f*h,this._w=c*u*h-d*f*g;break;case"ZYX":this._x=d*u*h-c*f*g,this._y=c*f*h+d*u*g,this._z=c*u*g-d*f*h,this._w=c*u*h+d*f*g;break;case"YZX":this._x=d*u*h+c*f*g,this._y=c*f*h+d*u*g,this._z=c*u*g-d*f*h,this._w=c*u*h-d*f*g;break;case"XZY":this._x=d*u*h-c*f*g,this._y=c*f*h-d*u*g,this._z=c*u*g+d*f*h,this._w=c*u*h+d*f*g;break;default:console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: "+o)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const i=t/2,s=Math.sin(i);return this._x=e.x*s,this._y=e.y*s,this._z=e.z*s,this._w=Math.cos(i),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,i=t[0],s=t[4],r=t[8],o=t[1],a=t[5],l=t[9],c=t[2],u=t[6],h=t[10],d=i+a+h;if(d>0){const f=.5/Math.sqrt(d+1);this._w=.25/f,this._x=(u-l)*f,this._y=(r-c)*f,this._z=(o-s)*f}else if(i>a&&i>h){const f=2*Math.sqrt(1+i-a-h);this._w=(u-l)/f,this._x=.25*f,this._y=(s+o)/f,this._z=(r+c)/f}else if(a>h){const f=2*Math.sqrt(1+a-i-h);this._w=(r-c)/f,this._x=(s+o)/f,this._y=.25*f,this._z=(l+u)/f}else{const f=2*Math.sqrt(1+h-i-a);this._w=(o-s)/f,this._x=(r+c)/f,this._y=(l+u)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let i=e.dot(t)+1;return i<1e-8?(i=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=i):(this._x=0,this._y=-e.z,this._z=e.y,this._w=i)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=i),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(Ze(this.dot(e),-1,1)))}rotateTowards(e,t){const i=this.angleTo(e);if(i===0)return this;const s=Math.min(1,t/i);return this.slerp(e,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const i=e._x,s=e._y,r=e._z,o=e._w,a=t._x,l=t._y,c=t._z,u=t._w;return this._x=i*u+o*a+s*c-r*l,this._y=s*u+o*l+r*a-i*c,this._z=r*u+o*c+i*l-s*a,this._w=o*u-i*a-s*l-r*c,this._onChangeCallback(),this}slerp(e,t){if(t===0)return this;if(t===1)return this.copy(e);const i=this._x,s=this._y,r=this._z,o=this._w;let a=o*e._w+i*e._x+s*e._y+r*e._z;if(a<0?(this._w=-e._w,this._x=-e._x,this._y=-e._y,this._z=-e._z,a=-a):this.copy(e),a>=1)return this._w=o,this._x=i,this._y=s,this._z=r,this;const l=1-a*a;if(l<=Number.EPSILON){const f=1-t;return this._w=f*o+t*this._w,this._x=f*i+t*this._x,this._y=f*s+t*this._y,this._z=f*r+t*this._z,this.normalize(),this}const c=Math.sqrt(l),u=Math.atan2(c,a),h=Math.sin((1-t)*u)/c,d=Math.sin(t*u)/c;return this._w=o*h+this._w*d,this._x=i*h+this._x*d,this._y=s*h+this._y*d,this._z=r*h+this._z*d,this._onChangeCallback(),this}slerpQuaternions(e,t,i){return this.copy(e).slerp(t,i)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),i=Math.random(),s=Math.sqrt(1-i),r=Math.sqrt(i);return this.set(s*Math.sin(e),s*Math.cos(e),r*Math.sin(t),r*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class R{constructor(e=0,t=0,i=0){R.prototype.isVector3=!0,this.x=e,this.y=t,this.z=i}set(e,t,i){return i===void 0&&(i=this.z),this.x=e,this.y=t,this.z=i,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Yl.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Yl.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[3]*i+r[6]*s,this.y=r[1]*t+r[4]*i+r[7]*s,this.z=r[2]*t+r[5]*i+r[8]*s,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,i=this.y,s=this.z,r=e.elements,o=1/(r[3]*t+r[7]*i+r[11]*s+r[15]);return this.x=(r[0]*t+r[4]*i+r[8]*s+r[12])*o,this.y=(r[1]*t+r[5]*i+r[9]*s+r[13])*o,this.z=(r[2]*t+r[6]*i+r[10]*s+r[14])*o,this}applyQuaternion(e){const t=this.x,i=this.y,s=this.z,r=e.x,o=e.y,a=e.z,l=e.w,c=2*(o*s-a*i),u=2*(a*t-r*s),h=2*(r*i-o*t);return this.x=t+l*c+o*h-a*u,this.y=i+l*u+a*c-r*h,this.z=s+l*h+r*u-o*c,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,i=this.y,s=this.z,r=e.elements;return this.x=r[0]*t+r[4]*i+r[8]*s,this.y=r[1]*t+r[5]*i+r[9]*s,this.z=r[2]*t+r[6]*i+r[10]*s,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=Ze(this.x,e.x,t.x),this.y=Ze(this.y,e.y,t.y),this.z=Ze(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=Ze(this.x,e,t),this.y=Ze(this.y,e,t),this.z=Ze(this.z,e,t),this}clampLength(e,t){const i=this.length();return this.divideScalar(i||1).multiplyScalar(Ze(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this.z=e.z+(t.z-e.z)*i,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const i=e.x,s=e.y,r=e.z,o=t.x,a=t.y,l=t.z;return this.x=s*l-r*a,this.y=r*o-i*l,this.z=i*a-s*o,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const i=e.dot(this)/t;return this.copy(e).multiplyScalar(i)}projectOnPlane(e){return Uo.copy(this).projectOnVector(e),this.sub(Uo)}reflect(e){return this.sub(Uo.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const i=this.dot(e)/t;return Math.acos(Ze(i,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,i=this.y-e.y,s=this.z-e.z;return t*t+i*i+s*s}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,i){const s=Math.sin(t)*e;return this.x=s*Math.sin(i),this.y=Math.cos(t)*e,this.z=s*Math.cos(i),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,i){return this.x=e*Math.sin(t),this.y=i,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),i=this.setFromMatrixColumn(e,1).length(),s=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=i,this.z=s,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,i=Math.sqrt(1-t*t);return this.x=i*Math.cos(e),this.y=t,this.z=i*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Uo=new R,Yl=new Ti;class Ve{constructor(e,t,i,s,r,o,a,l,c){Ve.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,i,s,r,o,a,l,c)}set(e,t,i,s,r,o,a,l,c){const u=this.elements;return u[0]=e,u[1]=s,u[2]=a,u[3]=t,u[4]=r,u[5]=l,u[6]=i,u[7]=o,u[8]=c,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,i=e.elements;return t[0]=i[0],t[1]=i[1],t[2]=i[2],t[3]=i[3],t[4]=i[4],t[5]=i[5],t[6]=i[6],t[7]=i[7],t[8]=i[8],this}extractBasis(e,t,i){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),i.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const i=e.elements,s=t.elements,r=this.elements,o=i[0],a=i[3],l=i[6],c=i[1],u=i[4],h=i[7],d=i[2],f=i[5],g=i[8],_=s[0],m=s[3],p=s[6],C=s[1],M=s[4],x=s[7],P=s[2],w=s[5],T=s[8];return r[0]=o*_+a*C+l*P,r[3]=o*m+a*M+l*w,r[6]=o*p+a*x+l*T,r[1]=c*_+u*C+h*P,r[4]=c*m+u*M+h*w,r[7]=c*p+u*x+h*T,r[2]=d*_+f*C+g*P,r[5]=d*m+f*M+g*w,r[8]=d*p+f*x+g*T,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8];return t*o*u-t*a*c-i*r*u+i*a*l+s*r*c-s*o*l}invert(){const e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],h=u*o-a*c,d=a*l-u*r,f=c*r-o*l,g=t*h+i*d+s*f;if(g===0)return this.set(0,0,0,0,0,0,0,0,0);const _=1/g;return e[0]=h*_,e[1]=(s*c-u*i)*_,e[2]=(a*i-s*o)*_,e[3]=d*_,e[4]=(u*t-s*l)*_,e[5]=(s*r-a*t)*_,e[6]=f*_,e[7]=(i*l-c*t)*_,e[8]=(o*t-i*r)*_,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,i,s,r,o,a){const l=Math.cos(r),c=Math.sin(r);return this.set(i*l,i*c,-i*(l*o+c*a)+o+e,-s*c,s*l,-s*(-c*o+l*a)+a+t,0,0,1),this}scale(e,t){return this.premultiply(Oo.makeScale(e,t)),this}rotate(e){return this.premultiply(Oo.makeRotation(-e)),this}translate(e,t){return this.premultiply(Oo.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),i=Math.sin(e);return this.set(t,-i,0,i,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,i=e.elements;for(let s=0;s<9;s++)if(t[s]!==i[s])return!1;return!0}fromArray(e,t=0){for(let i=0;i<9;i++)this.elements[i]=e[i+t];return this}toArray(e=[],t=0){const i=this.elements;return e[t]=i[0],e[t+1]=i[1],e[t+2]=i[2],e[t+3]=i[3],e[t+4]=i[4],e[t+5]=i[5],e[t+6]=i[6],e[t+7]=i[7],e[t+8]=i[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const Oo=new Ve;function Kh(n){for(let e=n.length-1;e>=0;--e)if(n[e]>=65535)return!0;return!1}function Ws(n){return document.createElementNS("http://www.w3.org/1999/xhtml",n)}function Qg(){const n=Ws("canvas");return n.style.display="block",n}const ql={};function Xs(n){n in ql||(ql[n]=!0,console.warn(n))}function e_(n,e,t){return new Promise(function(i,s){function r(){switch(n.clientWaitSync(e,n.SYNC_FLUSH_COMMANDS_BIT,0)){case n.WAIT_FAILED:s();break;case n.TIMEOUT_EXPIRED:setTimeout(r,t);break;default:i()}}setTimeout(r,t)})}const jl=new Ve().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Zl=new Ve().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function t_(){const n={enabled:!0,workingColorSpace:us,spaces:{},convert:function(s,r,o){return this.enabled===!1||r===o||!r||!o||(this.spaces[r].transfer===rt&&(s.r=$n(s.r),s.g=$n(s.g),s.b=$n(s.b)),this.spaces[r].primaries!==this.spaces[o].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[o].fromXYZ)),this.spaces[o].transfer===rt&&(s.r=is(s.r),s.g=is(s.g),s.b=is(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===ti?oo:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,o){return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[o].fromXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(s,r){return Xs("THREE.ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),n.workingToColorSpace(s,r)},toWorkingColorSpace:function(s,r){return Xs("THREE.ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),n.colorSpaceToWorking(s,r)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],i=[.3127,.329];return n.define({[us]:{primaries:e,whitePoint:i,transfer:oo,toXYZ:jl,fromXYZ:Zl,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:Ot},outputColorSpaceConfig:{drawingBufferColorSpace:Ot}},[Ot]:{primaries:e,whitePoint:i,transfer:rt,toXYZ:jl,fromXYZ:Zl,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:Ot}}}),n}const tt=t_();function $n(n){return n<.04045?n*.0773993808:Math.pow(n*.9478672986+.0521327014,2.4)}function is(n){return n<.0031308?n*12.92:1.055*Math.pow(n,.41666)-.055}let Ii;class n_{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let i;if(e instanceof HTMLCanvasElement)i=e;else{Ii===void 0&&(Ii=Ws("canvas")),Ii.width=e.width,Ii.height=e.height;const s=Ii.getContext("2d");e instanceof ImageData?s.putImageData(e,0,0):s.drawImage(e,0,0,e.width,e.height),i=Ii}return i.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=Ws("canvas");t.width=e.width,t.height=e.height;const i=t.getContext("2d");i.drawImage(e,0,0,e.width,e.height);const s=i.getImageData(0,0,e.width,e.height),r=s.data;for(let o=0;o<r.length;o++)r[o]=$n(r[o]/255)*255;return i.putImageData(s,0,0),t}else if(e.data){const t=e.data.slice(0);for(let i=0;i<t.length;i++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[i]=Math.floor($n(t[i]/255)*255):t[i]=$n(t[i]);return{data:t,width:e.width,height:e.height}}else return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let i_=0;class Bc{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:i_++}),this.uuid=Xn(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const i={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let o=0,a=s.length;o<a;o++)s[o].isDataTexture?r.push(Fo(s[o].image)):r.push(Fo(s[o]))}else r=Fo(s);i.url=r}return t||(e.images[this.uuid]=i),i}}function Fo(n){return typeof HTMLImageElement<"u"&&n instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&n instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&n instanceof ImageBitmap?n_.getDataURL(n):n.data?{data:Array.from(n.data),width:n.width,height:n.height,type:n.data.constructor.name}:(console.warn("THREE.Texture: Unable to serialize Texture."),{})}let s_=0;const ko=new R;class Vt extends Ri{constructor(e=Vt.DEFAULT_IMAGE,t=Vt.DEFAULT_MAPPING,i=Mi,s=Mi,r=Tn,o=wi,a=vn,l=Rn,c=Vt.DEFAULT_ANISOTROPY,u=ti){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:s_++}),this.uuid=Xn(),this.name="",this.source=new Bc(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=i,this.wrapT=s,this.magFilter=r,this.minFilter=o,this.anisotropy=c,this.format=a,this.internalFormat=null,this.type=l,this.offset=new ue(0,0),this.repeat=new ue(1,1),this.center=new ue(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Ve,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(ko).x}get height(){return this.source.getSize(ko).y}get depth(){return this.source.getSize(ko).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const i=e[t];if(i===void 0){console.warn(`THREE.Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){console.warn(`THREE.Texture.setValues(): property '${t}' does not exist.`);continue}s&&i&&s.isVector2&&i.isVector2||s&&i&&s.isVector3&&i.isVector3||s&&i&&s.isMatrix3&&i.isMatrix3?s.copy(i):this[t]=i}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const i={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(i.userData=this.userData),t||(e.textures[this.uuid]=i),i}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==Gh)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case Ga:e.x=e.x-Math.floor(e.x);break;case Mi:e.x=e.x<0?0:1;break;case Va:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case Ga:e.y=e.y-Math.floor(e.y);break;case Mi:e.y=e.y<0?0:1;break;case Va:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}Vt.DEFAULT_IMAGE=null;Vt.DEFAULT_MAPPING=Gh;Vt.DEFAULT_ANISOTROPY=1;class ct{constructor(e=0,t=0,i=0,s=1){ct.prototype.isVector4=!0,this.x=e,this.y=t,this.z=i,this.w=s}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,i,s){return this.x=e,this.y=t,this.z=i,this.w=s,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,i=this.y,s=this.z,r=this.w,o=e.elements;return this.x=o[0]*t+o[4]*i+o[8]*s+o[12]*r,this.y=o[1]*t+o[5]*i+o[9]*s+o[13]*r,this.z=o[2]*t+o[6]*i+o[10]*s+o[14]*r,this.w=o[3]*t+o[7]*i+o[11]*s+o[15]*r,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,i,s,r;const l=e.elements,c=l[0],u=l[4],h=l[8],d=l[1],f=l[5],g=l[9],_=l[2],m=l[6],p=l[10];if(Math.abs(u-d)<.01&&Math.abs(h-_)<.01&&Math.abs(g-m)<.01){if(Math.abs(u+d)<.1&&Math.abs(h+_)<.1&&Math.abs(g+m)<.1&&Math.abs(c+f+p-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const M=(c+1)/2,x=(f+1)/2,P=(p+1)/2,w=(u+d)/4,T=(h+_)/4,L=(g+m)/4;return M>x&&M>P?M<.01?(i=0,s=.707106781,r=.707106781):(i=Math.sqrt(M),s=w/i,r=T/i):x>P?x<.01?(i=.707106781,s=0,r=.707106781):(s=Math.sqrt(x),i=w/s,r=L/s):P<.01?(i=.707106781,s=.707106781,r=0):(r=Math.sqrt(P),i=T/r,s=L/r),this.set(i,s,r,t),this}let C=Math.sqrt((m-g)*(m-g)+(h-_)*(h-_)+(d-u)*(d-u));return Math.abs(C)<.001&&(C=1),this.x=(m-g)/C,this.y=(h-_)/C,this.z=(d-u)/C,this.w=Math.acos((c+f+p-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=Ze(this.x,e.x,t.x),this.y=Ze(this.y,e.y,t.y),this.z=Ze(this.z,e.z,t.z),this.w=Ze(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=Ze(this.x,e,t),this.y=Ze(this.y,e,t),this.z=Ze(this.z,e,t),this.w=Ze(this.w,e,t),this}clampLength(e,t){const i=this.length();return this.divideScalar(i||1).multiplyScalar(Ze(i,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,i){return this.x=e.x+(t.x-e.x)*i,this.y=e.y+(t.y-e.y)*i,this.z=e.z+(t.z-e.z)*i,this.w=e.w+(t.w-e.w)*i,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class r_ extends Ri{constructor(e=1,t=1,i={}){super(),i=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:Tn,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},i),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=i.depth,this.scissor=new ct(0,0,e,t),this.scissorTest=!1,this.viewport=new ct(0,0,e,t);const s={width:e,height:t,depth:i.depth},r=new Vt(s);this.textures=[];const o=i.count;for(let a=0;a<o;a++)this.textures[a]=r.clone(),this.textures[a].isRenderTargetTexture=!0,this.textures[a].renderTarget=this;this._setTextureOptions(i),this.depthBuffer=i.depthBuffer,this.stencilBuffer=i.stencilBuffer,this.resolveDepthBuffer=i.resolveDepthBuffer,this.resolveStencilBuffer=i.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=i.depthTexture,this.samples=i.samples,this.multiview=i.multiview}_setTextureOptions(e={}){const t={minFilter:Tn,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let i=0;i<this.textures.length;i++)this.textures[i].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,i=1){if(this.width!==e||this.height!==t||this.depth!==i){this.width=e,this.height=t,this.depth=i;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=e,this.textures[s].image.height=t,this.textures[s].image.depth=i,this.textures[s].isArrayTexture=this.textures[s].image.depth>1;this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,i=e.textures.length;t<i;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const s=Object.assign({},e.textures[t].image);this.textures[t].source=new Bc(s)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class Ai extends r_{constructor(e=1,t=1,i={}){super(e,t,i),this.isWebGLRenderTarget=!0}}class Qh extends Vt{constructor(e=null,t=1,i=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:i,depth:s},this.magFilter=xn,this.minFilter=xn,this.wrapR=Mi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class o_ extends Vt{constructor(e=null,t=1,i=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:i,depth:s},this.magFilter=xn,this.minFilter=xn,this.wrapR=Mi,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Qs{constructor(e=new R(1/0,1/0,1/0),t=new R(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,i=e.length;t<i;t+=3)this.expandByPoint(fn.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,i=e.count;t<i;t++)this.expandByPoint(fn.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,i=e.length;t<i;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const i=fn.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(i),this.max.copy(e).add(i),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const i=e.geometry;if(i!==void 0){const r=i.getAttribute("position");if(t===!0&&r!==void 0&&e.isInstancedMesh!==!0)for(let o=0,a=r.count;o<a;o++)e.isMesh===!0?e.getVertexPosition(o,fn):fn.fromBufferAttribute(r,o),fn.applyMatrix4(e.matrixWorld),this.expandByPoint(fn);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),fr.copy(e.boundingBox)):(i.boundingBox===null&&i.computeBoundingBox(),fr.copy(i.boundingBox)),fr.applyMatrix4(e.matrixWorld),this.union(fr)}const s=e.children;for(let r=0,o=s.length;r<o;r++)this.expandByObject(s[r],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,fn),fn.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,i;return e.normal.x>0?(t=e.normal.x*this.min.x,i=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,i=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,i+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,i+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,i+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,i+=e.normal.z*this.min.z),t<=-e.constant&&i>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(bs),pr.subVectors(this.max,bs),Ni.subVectors(e.a,bs),Ui.subVectors(e.b,bs),Oi.subVectors(e.c,bs),qn.subVectors(Ui,Ni),jn.subVectors(Oi,Ui),hi.subVectors(Ni,Oi);let t=[0,-qn.z,qn.y,0,-jn.z,jn.y,0,-hi.z,hi.y,qn.z,0,-qn.x,jn.z,0,-jn.x,hi.z,0,-hi.x,-qn.y,qn.x,0,-jn.y,jn.x,0,-hi.y,hi.x,0];return!zo(t,Ni,Ui,Oi,pr)||(t=[1,0,0,0,1,0,0,0,1],!zo(t,Ni,Ui,Oi,pr))?!1:(mr.crossVectors(qn,jn),t=[mr.x,mr.y,mr.z],zo(t,Ni,Ui,Oi,pr))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,fn).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(fn).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Un[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Un[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Un[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Un[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Un[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Un[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Un[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Un[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Un),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Un=[new R,new R,new R,new R,new R,new R,new R,new R],fn=new R,fr=new Qs,Ni=new R,Ui=new R,Oi=new R,qn=new R,jn=new R,hi=new R,bs=new R,pr=new R,mr=new R,di=new R;function zo(n,e,t,i,s){for(let r=0,o=n.length-3;r<=o;r+=3){di.fromArray(n,r);const a=s.x*Math.abs(di.x)+s.y*Math.abs(di.y)+s.z*Math.abs(di.z),l=e.dot(di),c=t.dot(di),u=i.dot(di);if(Math.max(-Math.max(l,c,u),Math.min(l,c,u))>a)return!1}return!0}const a_=new Qs,Ms=new R,Bo=new R;class xo{constructor(e=new R,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const i=this.center;t!==void 0?i.copy(t):a_.setFromPoints(e).getCenter(i);let s=0;for(let r=0,o=e.length;r<o;r++)s=Math.max(s,i.distanceToSquared(e[r]));return this.radius=Math.sqrt(s),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const i=this.center.distanceToSquared(e);return t.copy(e),i>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;Ms.subVectors(e,this.center);const t=Ms.lengthSq();if(t>this.radius*this.radius){const i=Math.sqrt(t),s=(i-this.radius)*.5;this.center.addScaledVector(Ms,s/i),this.radius+=s}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(Bo.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(Ms.copy(e.center).add(Bo)),this.expandByPoint(Ms.copy(e.center).sub(Bo))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}const On=new R,Ho=new R,gr=new R,Zn=new R,Go=new R,_r=new R,Vo=new R;class yo{constructor(e=new R,t=new R(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,On)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const i=t.dot(this.direction);return i<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,i)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=On.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(On.copy(this.origin).addScaledVector(this.direction,t),On.distanceToSquared(e))}distanceSqToSegment(e,t,i,s){Ho.copy(e).add(t).multiplyScalar(.5),gr.copy(t).sub(e).normalize(),Zn.copy(this.origin).sub(Ho);const r=e.distanceTo(t)*.5,o=-this.direction.dot(gr),a=Zn.dot(this.direction),l=-Zn.dot(gr),c=Zn.lengthSq(),u=Math.abs(1-o*o);let h,d,f,g;if(u>0)if(h=o*l-a,d=o*a-l,g=r*u,h>=0)if(d>=-g)if(d<=g){const _=1/u;h*=_,d*=_,f=h*(h+o*d+2*a)+d*(o*h+d+2*l)+c}else d=r,h=Math.max(0,-(o*d+a)),f=-h*h+d*(d+2*l)+c;else d=-r,h=Math.max(0,-(o*d+a)),f=-h*h+d*(d+2*l)+c;else d<=-g?(h=Math.max(0,-(-o*r+a)),d=h>0?-r:Math.min(Math.max(-r,-l),r),f=-h*h+d*(d+2*l)+c):d<=g?(h=0,d=Math.min(Math.max(-r,-l),r),f=d*(d+2*l)+c):(h=Math.max(0,-(o*r+a)),d=h>0?r:Math.min(Math.max(-r,-l),r),f=-h*h+d*(d+2*l)+c);else d=o>0?-r:r,h=Math.max(0,-(o*d+a)),f=-h*h+d*(d+2*l)+c;return i&&i.copy(this.origin).addScaledVector(this.direction,h),s&&s.copy(Ho).addScaledVector(gr,d),f}intersectSphere(e,t){On.subVectors(e.center,this.origin);const i=On.dot(this.direction),s=On.dot(On)-i*i,r=e.radius*e.radius;if(s>r)return null;const o=Math.sqrt(r-s),a=i-o,l=i+o;return l<0?null:a<0?this.at(l,t):this.at(a,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const i=-(this.origin.dot(e.normal)+e.constant)/t;return i>=0?i:null}intersectPlane(e,t){const i=this.distanceToPlane(e);return i===null?null:this.at(i,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let i,s,r,o,a,l;const c=1/this.direction.x,u=1/this.direction.y,h=1/this.direction.z,d=this.origin;return c>=0?(i=(e.min.x-d.x)*c,s=(e.max.x-d.x)*c):(i=(e.max.x-d.x)*c,s=(e.min.x-d.x)*c),u>=0?(r=(e.min.y-d.y)*u,o=(e.max.y-d.y)*u):(r=(e.max.y-d.y)*u,o=(e.min.y-d.y)*u),i>o||r>s||((r>i||isNaN(i))&&(i=r),(o<s||isNaN(s))&&(s=o),h>=0?(a=(e.min.z-d.z)*h,l=(e.max.z-d.z)*h):(a=(e.max.z-d.z)*h,l=(e.min.z-d.z)*h),i>l||a>s)||((a>i||i!==i)&&(i=a),(l<s||s!==s)&&(s=l),s<0)?null:this.at(i>=0?i:s,t)}intersectsBox(e){return this.intersectBox(e,On)!==null}intersectTriangle(e,t,i,s,r){Go.subVectors(t,e),_r.subVectors(i,e),Vo.crossVectors(Go,_r);let o=this.direction.dot(Vo),a;if(o>0){if(s)return null;a=1}else if(o<0)a=-1,o=-o;else return null;Zn.subVectors(this.origin,e);const l=a*this.direction.dot(_r.crossVectors(Zn,_r));if(l<0)return null;const c=a*this.direction.dot(Go.cross(Zn));if(c<0||l+c>o)return null;const u=-a*Zn.dot(Vo);return u<0?null:this.at(u/o,r)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class ft{constructor(e,t,i,s,r,o,a,l,c,u,h,d,f,g,_,m){ft.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,i,s,r,o,a,l,c,u,h,d,f,g,_,m)}set(e,t,i,s,r,o,a,l,c,u,h,d,f,g,_,m){const p=this.elements;return p[0]=e,p[4]=t,p[8]=i,p[12]=s,p[1]=r,p[5]=o,p[9]=a,p[13]=l,p[2]=c,p[6]=u,p[10]=h,p[14]=d,p[3]=f,p[7]=g,p[11]=_,p[15]=m,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new ft().fromArray(this.elements)}copy(e){const t=this.elements,i=e.elements;return t[0]=i[0],t[1]=i[1],t[2]=i[2],t[3]=i[3],t[4]=i[4],t[5]=i[5],t[6]=i[6],t[7]=i[7],t[8]=i[8],t[9]=i[9],t[10]=i[10],t[11]=i[11],t[12]=i[12],t[13]=i[13],t[14]=i[14],t[15]=i[15],this}copyPosition(e){const t=this.elements,i=e.elements;return t[12]=i[12],t[13]=i[13],t[14]=i[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,i){return e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),i.setFromMatrixColumn(this,2),this}makeBasis(e,t,i){return this.set(e.x,t.x,i.x,0,e.y,t.y,i.y,0,e.z,t.z,i.z,0,0,0,0,1),this}extractRotation(e){const t=this.elements,i=e.elements,s=1/Fi.setFromMatrixColumn(e,0).length(),r=1/Fi.setFromMatrixColumn(e,1).length(),o=1/Fi.setFromMatrixColumn(e,2).length();return t[0]=i[0]*s,t[1]=i[1]*s,t[2]=i[2]*s,t[3]=0,t[4]=i[4]*r,t[5]=i[5]*r,t[6]=i[6]*r,t[7]=0,t[8]=i[8]*o,t[9]=i[9]*o,t[10]=i[10]*o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,i=e.x,s=e.y,r=e.z,o=Math.cos(i),a=Math.sin(i),l=Math.cos(s),c=Math.sin(s),u=Math.cos(r),h=Math.sin(r);if(e.order==="XYZ"){const d=o*u,f=o*h,g=a*u,_=a*h;t[0]=l*u,t[4]=-l*h,t[8]=c,t[1]=f+g*c,t[5]=d-_*c,t[9]=-a*l,t[2]=_-d*c,t[6]=g+f*c,t[10]=o*l}else if(e.order==="YXZ"){const d=l*u,f=l*h,g=c*u,_=c*h;t[0]=d+_*a,t[4]=g*a-f,t[8]=o*c,t[1]=o*h,t[5]=o*u,t[9]=-a,t[2]=f*a-g,t[6]=_+d*a,t[10]=o*l}else if(e.order==="ZXY"){const d=l*u,f=l*h,g=c*u,_=c*h;t[0]=d-_*a,t[4]=-o*h,t[8]=g+f*a,t[1]=f+g*a,t[5]=o*u,t[9]=_-d*a,t[2]=-o*c,t[6]=a,t[10]=o*l}else if(e.order==="ZYX"){const d=o*u,f=o*h,g=a*u,_=a*h;t[0]=l*u,t[4]=g*c-f,t[8]=d*c+_,t[1]=l*h,t[5]=_*c+d,t[9]=f*c-g,t[2]=-c,t[6]=a*l,t[10]=o*l}else if(e.order==="YZX"){const d=o*l,f=o*c,g=a*l,_=a*c;t[0]=l*u,t[4]=_-d*h,t[8]=g*h+f,t[1]=h,t[5]=o*u,t[9]=-a*u,t[2]=-c*u,t[6]=f*h+g,t[10]=d-_*h}else if(e.order==="XZY"){const d=o*l,f=o*c,g=a*l,_=a*c;t[0]=l*u,t[4]=-h,t[8]=c*u,t[1]=d*h+_,t[5]=o*u,t[9]=f*h-g,t[2]=g*h-f,t[6]=a*u,t[10]=_*h+d}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(c_,e,l_)}lookAt(e,t,i){const s=this.elements;return tn.subVectors(e,t),tn.lengthSq()===0&&(tn.z=1),tn.normalize(),Jn.crossVectors(i,tn),Jn.lengthSq()===0&&(Math.abs(i.z)===1?tn.x+=1e-4:tn.z+=1e-4,tn.normalize(),Jn.crossVectors(i,tn)),Jn.normalize(),vr.crossVectors(tn,Jn),s[0]=Jn.x,s[4]=vr.x,s[8]=tn.x,s[1]=Jn.y,s[5]=vr.y,s[9]=tn.y,s[2]=Jn.z,s[6]=vr.z,s[10]=tn.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const i=e.elements,s=t.elements,r=this.elements,o=i[0],a=i[4],l=i[8],c=i[12],u=i[1],h=i[5],d=i[9],f=i[13],g=i[2],_=i[6],m=i[10],p=i[14],C=i[3],M=i[7],x=i[11],P=i[15],w=s[0],T=s[4],L=s[8],y=s[12],b=s[1],N=s[5],O=s[9],G=s[13],V=s[2],X=s[6],q=s[10],J=s[14],z=s[3],H=s[7],D=s[11],W=s[15];return r[0]=o*w+a*b+l*V+c*z,r[4]=o*T+a*N+l*X+c*H,r[8]=o*L+a*O+l*q+c*D,r[12]=o*y+a*G+l*J+c*W,r[1]=u*w+h*b+d*V+f*z,r[5]=u*T+h*N+d*X+f*H,r[9]=u*L+h*O+d*q+f*D,r[13]=u*y+h*G+d*J+f*W,r[2]=g*w+_*b+m*V+p*z,r[6]=g*T+_*N+m*X+p*H,r[10]=g*L+_*O+m*q+p*D,r[14]=g*y+_*G+m*J+p*W,r[3]=C*w+M*b+x*V+P*z,r[7]=C*T+M*N+x*X+P*H,r[11]=C*L+M*O+x*q+P*D,r[15]=C*y+M*G+x*J+P*W,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],i=e[4],s=e[8],r=e[12],o=e[1],a=e[5],l=e[9],c=e[13],u=e[2],h=e[6],d=e[10],f=e[14],g=e[3],_=e[7],m=e[11],p=e[15];return g*(+r*l*h-s*c*h-r*a*d+i*c*d+s*a*f-i*l*f)+_*(+t*l*f-t*c*d+r*o*d-s*o*f+s*c*u-r*l*u)+m*(+t*c*h-t*a*f-r*o*h+i*o*f+r*a*u-i*c*u)+p*(-s*a*u-t*l*h+t*a*d+s*o*h-i*o*d+i*l*u)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,i){const s=this.elements;return e.isVector3?(s[12]=e.x,s[13]=e.y,s[14]=e.z):(s[12]=e,s[13]=t,s[14]=i),this}invert(){const e=this.elements,t=e[0],i=e[1],s=e[2],r=e[3],o=e[4],a=e[5],l=e[6],c=e[7],u=e[8],h=e[9],d=e[10],f=e[11],g=e[12],_=e[13],m=e[14],p=e[15],C=h*m*c-_*d*c+_*l*f-a*m*f-h*l*p+a*d*p,M=g*d*c-u*m*c-g*l*f+o*m*f+u*l*p-o*d*p,x=u*_*c-g*h*c+g*a*f-o*_*f-u*a*p+o*h*p,P=g*h*l-u*_*l-g*a*d+o*_*d+u*a*m-o*h*m,w=t*C+i*M+s*x+r*P;if(w===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const T=1/w;return e[0]=C*T,e[1]=(_*d*r-h*m*r-_*s*f+i*m*f+h*s*p-i*d*p)*T,e[2]=(a*m*r-_*l*r+_*s*c-i*m*c-a*s*p+i*l*p)*T,e[3]=(h*l*r-a*d*r-h*s*c+i*d*c+a*s*f-i*l*f)*T,e[4]=M*T,e[5]=(u*m*r-g*d*r+g*s*f-t*m*f-u*s*p+t*d*p)*T,e[6]=(g*l*r-o*m*r-g*s*c+t*m*c+o*s*p-t*l*p)*T,e[7]=(o*d*r-u*l*r+u*s*c-t*d*c-o*s*f+t*l*f)*T,e[8]=x*T,e[9]=(g*h*r-u*_*r-g*i*f+t*_*f+u*i*p-t*h*p)*T,e[10]=(o*_*r-g*a*r+g*i*c-t*_*c-o*i*p+t*a*p)*T,e[11]=(u*a*r-o*h*r-u*i*c+t*h*c+o*i*f-t*a*f)*T,e[12]=P*T,e[13]=(u*_*s-g*h*s+g*i*d-t*_*d-u*i*m+t*h*m)*T,e[14]=(g*a*s-o*_*s-g*i*l+t*_*l+o*i*m-t*a*m)*T,e[15]=(o*h*s-u*a*s+u*i*l-t*h*l-o*i*d+t*a*d)*T,this}scale(e){const t=this.elements,i=e.x,s=e.y,r=e.z;return t[0]*=i,t[4]*=s,t[8]*=r,t[1]*=i,t[5]*=s,t[9]*=r,t[2]*=i,t[6]*=s,t[10]*=r,t[3]*=i,t[7]*=s,t[11]*=r,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],i=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],s=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,i,s))}makeTranslation(e,t,i){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,i,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),i=Math.sin(e);return this.set(1,0,0,0,0,t,-i,0,0,i,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),i=Math.sin(e);return this.set(t,0,i,0,0,1,0,0,-i,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),i=Math.sin(e);return this.set(t,-i,0,0,i,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const i=Math.cos(t),s=Math.sin(t),r=1-i,o=e.x,a=e.y,l=e.z,c=r*o,u=r*a;return this.set(c*o+i,c*a-s*l,c*l+s*a,0,c*a+s*l,u*a+i,u*l-s*o,0,c*l-s*a,u*l+s*o,r*l*l+i,0,0,0,0,1),this}makeScale(e,t,i){return this.set(e,0,0,0,0,t,0,0,0,0,i,0,0,0,0,1),this}makeShear(e,t,i,s,r,o){return this.set(1,i,r,0,e,1,o,0,t,s,1,0,0,0,0,1),this}compose(e,t,i){const s=this.elements,r=t._x,o=t._y,a=t._z,l=t._w,c=r+r,u=o+o,h=a+a,d=r*c,f=r*u,g=r*h,_=o*u,m=o*h,p=a*h,C=l*c,M=l*u,x=l*h,P=i.x,w=i.y,T=i.z;return s[0]=(1-(_+p))*P,s[1]=(f+x)*P,s[2]=(g-M)*P,s[3]=0,s[4]=(f-x)*w,s[5]=(1-(d+p))*w,s[6]=(m+C)*w,s[7]=0,s[8]=(g+M)*T,s[9]=(m-C)*T,s[10]=(1-(d+_))*T,s[11]=0,s[12]=e.x,s[13]=e.y,s[14]=e.z,s[15]=1,this}decompose(e,t,i){const s=this.elements;let r=Fi.set(s[0],s[1],s[2]).length();const o=Fi.set(s[4],s[5],s[6]).length(),a=Fi.set(s[8],s[9],s[10]).length();this.determinant()<0&&(r=-r),e.x=s[12],e.y=s[13],e.z=s[14],pn.copy(this);const c=1/r,u=1/o,h=1/a;return pn.elements[0]*=c,pn.elements[1]*=c,pn.elements[2]*=c,pn.elements[4]*=u,pn.elements[5]*=u,pn.elements[6]*=u,pn.elements[8]*=h,pn.elements[9]*=h,pn.elements[10]*=h,t.setFromRotationMatrix(pn),i.x=r,i.y=o,i.z=a,this}makePerspective(e,t,i,s,r,o,a=An,l=!1){const c=this.elements,u=2*r/(t-e),h=2*r/(i-s),d=(t+e)/(t-e),f=(i+s)/(i-s);let g,_;if(l)g=r/(o-r),_=o*r/(o-r);else if(a===An)g=-(o+r)/(o-r),_=-2*o*r/(o-r);else if(a===ao)g=-o/(o-r),_=-o*r/(o-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=d,c[12]=0,c[1]=0,c[5]=h,c[9]=f,c[13]=0,c[2]=0,c[6]=0,c[10]=g,c[14]=_,c[3]=0,c[7]=0,c[11]=-1,c[15]=0,this}makeOrthographic(e,t,i,s,r,o,a=An,l=!1){const c=this.elements,u=2/(t-e),h=2/(i-s),d=-(t+e)/(t-e),f=-(i+s)/(i-s);let g,_;if(l)g=1/(o-r),_=o/(o-r);else if(a===An)g=-2/(o-r),_=-(o+r)/(o-r);else if(a===ao)g=-1/(o-r),_=-r/(o-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+a);return c[0]=u,c[4]=0,c[8]=0,c[12]=d,c[1]=0,c[5]=h,c[9]=0,c[13]=f,c[2]=0,c[6]=0,c[10]=g,c[14]=_,c[3]=0,c[7]=0,c[11]=0,c[15]=1,this}equals(e){const t=this.elements,i=e.elements;for(let s=0;s<16;s++)if(t[s]!==i[s])return!1;return!0}fromArray(e,t=0){for(let i=0;i<16;i++)this.elements[i]=e[i+t];return this}toArray(e=[],t=0){const i=this.elements;return e[t]=i[0],e[t+1]=i[1],e[t+2]=i[2],e[t+3]=i[3],e[t+4]=i[4],e[t+5]=i[5],e[t+6]=i[6],e[t+7]=i[7],e[t+8]=i[8],e[t+9]=i[9],e[t+10]=i[10],e[t+11]=i[11],e[t+12]=i[12],e[t+13]=i[13],e[t+14]=i[14],e[t+15]=i[15],e}}const Fi=new R,pn=new ft,c_=new R(0,0,0),l_=new R(1,1,1),Jn=new R,vr=new R,tn=new R,Jl=new ft,Kl=new Ti;class Yn{constructor(e=0,t=0,i=0,s=Yn.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=i,this._order=s}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,i,s=this._order){return this._x=e,this._y=t,this._z=i,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,i=!0){const s=e.elements,r=s[0],o=s[4],a=s[8],l=s[1],c=s[5],u=s[9],h=s[2],d=s[6],f=s[10];switch(t){case"XYZ":this._y=Math.asin(Ze(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(-u,f),this._z=Math.atan2(-o,r)):(this._x=Math.atan2(d,c),this._z=0);break;case"YXZ":this._x=Math.asin(-Ze(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(a,f),this._z=Math.atan2(l,c)):(this._y=Math.atan2(-h,r),this._z=0);break;case"ZXY":this._x=Math.asin(Ze(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-h,f),this._z=Math.atan2(-o,c)):(this._y=0,this._z=Math.atan2(l,r));break;case"ZYX":this._y=Math.asin(-Ze(h,-1,1)),Math.abs(h)<.9999999?(this._x=Math.atan2(d,f),this._z=Math.atan2(l,r)):(this._x=0,this._z=Math.atan2(-o,c));break;case"YZX":this._z=Math.asin(Ze(l,-1,1)),Math.abs(l)<.9999999?(this._x=Math.atan2(-u,c),this._y=Math.atan2(-h,r)):(this._x=0,this._y=Math.atan2(a,f));break;case"XZY":this._z=Math.asin(-Ze(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(d,c),this._y=Math.atan2(a,r)):(this._x=Math.atan2(-u,f),this._y=0);break;default:console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,i===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,i){return Jl.makeRotationFromQuaternion(e),this.setFromRotationMatrix(Jl,t,i)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Kl.setFromEuler(this),this.setFromQuaternion(Kl,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Yn.DEFAULT_ORDER="XYZ";class Hc{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let u_=0;const Ql=new R,ki=new Ti,Fn=new ft,xr=new R,ws=new R,h_=new R,d_=new Ti,eu=new R(1,0,0),tu=new R(0,1,0),nu=new R(0,0,1),iu={type:"added"},f_={type:"removed"},zi={type:"childadded",child:null},Wo={type:"childremoved",child:null};class Lt extends Ri{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:u_++}),this.uuid=Xn(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Lt.DEFAULT_UP.clone();const e=new R,t=new Yn,i=new Ti,s=new R(1,1,1);function r(){i.setFromEuler(t,!1)}function o(){t.setFromQuaternion(i,void 0,!1)}t._onChange(r),i._onChange(o),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:i},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new ft},normalMatrix:{value:new Ve}}),this.matrix=new ft,this.matrixWorld=new ft,this.matrixAutoUpdate=Lt.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Lt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new Hc,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.userData={}}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return ki.setFromAxisAngle(e,t),this.quaternion.multiply(ki),this}rotateOnWorldAxis(e,t){return ki.setFromAxisAngle(e,t),this.quaternion.premultiply(ki),this}rotateX(e){return this.rotateOnAxis(eu,e)}rotateY(e){return this.rotateOnAxis(tu,e)}rotateZ(e){return this.rotateOnAxis(nu,e)}translateOnAxis(e,t){return Ql.copy(e).applyQuaternion(this.quaternion),this.position.add(Ql.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(eu,e)}translateY(e){return this.translateOnAxis(tu,e)}translateZ(e){return this.translateOnAxis(nu,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Fn.copy(this.matrixWorld).invert())}lookAt(e,t,i){e.isVector3?xr.copy(e):xr.set(e,t,i);const s=this.parent;this.updateWorldMatrix(!0,!1),ws.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Fn.lookAt(ws,xr,this.up):Fn.lookAt(xr,ws,this.up),this.quaternion.setFromRotationMatrix(Fn),s&&(Fn.extractRotation(s.matrixWorld),ki.setFromRotationMatrix(Fn),this.quaternion.premultiply(ki.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(console.error("THREE.Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(iu),zi.child=e,this.dispatchEvent(zi),zi.child=null):console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let i=0;i<arguments.length;i++)this.remove(arguments[i]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(f_),Wo.child=e,this.dispatchEvent(Wo),Wo.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Fn.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Fn.multiply(e.parent.matrixWorld)),e.applyMatrix4(Fn),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(iu),zi.child=e,this.dispatchEvent(zi),zi.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let i=0,s=this.children.length;i<s;i++){const o=this.children[i].getObjectByProperty(e,t);if(o!==void 0)return o}}getObjectsByProperty(e,t,i=[]){this[e]===t&&i.push(this);const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].getObjectsByProperty(e,t,i);return i}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(ws,e,h_),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(ws,d_,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale),this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let i=0,s=t.length;i<s;i++)t[i].updateMatrixWorld(e)}updateWorldMatrix(e,t){const i=this.parent;if(e===!0&&i!==null&&i.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const s=this.children;for(let r=0,o=s.length;r<o;r++)s[r].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",i={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},i.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const s={};s.uuid=this.uuid,s.type=this.type,this.name!==""&&(s.name=this.name),this.castShadow===!0&&(s.castShadow=!0),this.receiveShadow===!0&&(s.receiveShadow=!0),this.visible===!1&&(s.visible=!1),this.frustumCulled===!1&&(s.frustumCulled=!1),this.renderOrder!==0&&(s.renderOrder=this.renderOrder),Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.matrixAutoUpdate===!1&&(s.matrixAutoUpdate=!1),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(a=>({...a,boundingBox:a.boundingBox?a.boundingBox.toJSON():void 0,boundingSphere:a.boundingSphere?a.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(a=>({...a})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(e),s.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(a,l){return a[l.uuid]===void 0&&(a[l.uuid]=l.toJSON(e)),l.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(e.geometries,this.geometry);const a=this.geometry.parameters;if(a!==void 0&&a.shapes!==void 0){const l=a.shapes;if(Array.isArray(l))for(let c=0,u=l.length;c<u;c++){const h=l[c];r(e.shapes,h)}else r(e.shapes,l)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(e.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const a=[];for(let l=0,c=this.material.length;l<c;l++)a.push(r(e.materials,this.material[l]));s.material=a}else s.material=r(e.materials,this.material);if(this.children.length>0){s.children=[];for(let a=0;a<this.children.length;a++)s.children.push(this.children[a].toJSON(e).object)}if(this.animations.length>0){s.animations=[];for(let a=0;a<this.animations.length;a++){const l=this.animations[a];s.animations.push(r(e.animations,l))}}if(t){const a=o(e.geometries),l=o(e.materials),c=o(e.textures),u=o(e.images),h=o(e.shapes),d=o(e.skeletons),f=o(e.animations),g=o(e.nodes);a.length>0&&(i.geometries=a),l.length>0&&(i.materials=l),c.length>0&&(i.textures=c),u.length>0&&(i.images=u),h.length>0&&(i.shapes=h),d.length>0&&(i.skeletons=d),f.length>0&&(i.animations=f),g.length>0&&(i.nodes=g)}return i.object=s,i;function o(a){const l=[];for(const c in a){const u=a[c];delete u.metadata,l.push(u)}return l}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let i=0;i<e.children.length;i++){const s=e.children[i];this.add(s.clone())}return this}}Lt.DEFAULT_UP=new R(0,1,0);Lt.DEFAULT_MATRIX_AUTO_UPDATE=!0;Lt.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;const mn=new R,kn=new R,Xo=new R,zn=new R,Bi=new R,Hi=new R,su=new R,$o=new R,Yo=new R,qo=new R,jo=new ct,Zo=new ct,Jo=new ct;class ln{constructor(e=new R,t=new R,i=new R){this.a=e,this.b=t,this.c=i}static getNormal(e,t,i,s){s.subVectors(i,t),mn.subVectors(e,t),s.cross(mn);const r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(e,t,i,s,r){mn.subVectors(s,t),kn.subVectors(i,t),Xo.subVectors(e,t);const o=mn.dot(mn),a=mn.dot(kn),l=mn.dot(Xo),c=kn.dot(kn),u=kn.dot(Xo),h=o*c-a*a;if(h===0)return r.set(0,0,0),null;const d=1/h,f=(c*l-a*u)*d,g=(o*u-a*l)*d;return r.set(1-f-g,g,f)}static containsPoint(e,t,i,s){return this.getBarycoord(e,t,i,s,zn)===null?!1:zn.x>=0&&zn.y>=0&&zn.x+zn.y<=1}static getInterpolation(e,t,i,s,r,o,a,l){return this.getBarycoord(e,t,i,s,zn)===null?(l.x=0,l.y=0,"z"in l&&(l.z=0),"w"in l&&(l.w=0),null):(l.setScalar(0),l.addScaledVector(r,zn.x),l.addScaledVector(o,zn.y),l.addScaledVector(a,zn.z),l)}static getInterpolatedAttribute(e,t,i,s,r,o){return jo.setScalar(0),Zo.setScalar(0),Jo.setScalar(0),jo.fromBufferAttribute(e,t),Zo.fromBufferAttribute(e,i),Jo.fromBufferAttribute(e,s),o.setScalar(0),o.addScaledVector(jo,r.x),o.addScaledVector(Zo,r.y),o.addScaledVector(Jo,r.z),o}static isFrontFacing(e,t,i,s){return mn.subVectors(i,t),kn.subVectors(e,t),mn.cross(kn).dot(s)<0}set(e,t,i){return this.a.copy(e),this.b.copy(t),this.c.copy(i),this}setFromPointsAndIndices(e,t,i,s){return this.a.copy(e[t]),this.b.copy(e[i]),this.c.copy(e[s]),this}setFromAttributeAndIndices(e,t,i,s){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,i),this.c.fromBufferAttribute(e,s),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return mn.subVectors(this.c,this.b),kn.subVectors(this.a,this.b),mn.cross(kn).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return ln.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return ln.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,i,s,r){return ln.getInterpolation(e,this.a,this.b,this.c,t,i,s,r)}containsPoint(e){return ln.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return ln.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const i=this.a,s=this.b,r=this.c;let o,a;Bi.subVectors(s,i),Hi.subVectors(r,i),$o.subVectors(e,i);const l=Bi.dot($o),c=Hi.dot($o);if(l<=0&&c<=0)return t.copy(i);Yo.subVectors(e,s);const u=Bi.dot(Yo),h=Hi.dot(Yo);if(u>=0&&h<=u)return t.copy(s);const d=l*h-u*c;if(d<=0&&l>=0&&u<=0)return o=l/(l-u),t.copy(i).addScaledVector(Bi,o);qo.subVectors(e,r);const f=Bi.dot(qo),g=Hi.dot(qo);if(g>=0&&f<=g)return t.copy(r);const _=f*c-l*g;if(_<=0&&c>=0&&g<=0)return a=c/(c-g),t.copy(i).addScaledVector(Hi,a);const m=u*g-f*h;if(m<=0&&h-u>=0&&f-g>=0)return su.subVectors(r,s),a=(h-u)/(h-u+(f-g)),t.copy(s).addScaledVector(su,a);const p=1/(m+_+d);return o=_*p,a=d*p,t.copy(i).addScaledVector(Bi,o).addScaledVector(Hi,a)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}const ed={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Kn={h:0,s:0,l:0},yr={h:0,s:0,l:0};function Ko(n,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?n+(e-n)*6*t:t<1/2?e:t<2/3?n+(e-n)*6*(2/3-t):n}class Qe{constructor(e,t,i){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,i)}set(e,t,i){if(t===void 0&&i===void 0){const s=e;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(e,t,i);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=Ot){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,tt.colorSpaceToWorking(this,t),this}setRGB(e,t,i,s=tt.workingColorSpace){return this.r=e,this.g=t,this.b=i,tt.colorSpaceToWorking(this,s),this}setHSL(e,t,i,s=tt.workingColorSpace){if(e=Jg(e,1),t=Ze(t,0,1),i=Ze(i,0,1),t===0)this.r=this.g=this.b=i;else{const r=i<=.5?i*(1+t):i+t-i*t,o=2*i-r;this.r=Ko(o,r,e+1/3),this.g=Ko(o,r,e),this.b=Ko(o,r,e-1/3)}return tt.colorSpaceToWorking(this,s),this}setStyle(e,t=Ot){function i(r){r!==void 0&&parseFloat(r)<1&&console.warn("THREE.Color: Alpha component of "+e+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(e)){let r;const o=s[1],a=s[2];switch(o){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,t);if(r=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setRGB(Math.min(100,parseInt(r[1],10))/100,Math.min(100,parseInt(r[2],10))/100,Math.min(100,parseInt(r[3],10))/100,t);break;case"hsl":case"hsla":if(r=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(a))return i(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,t);break;default:console.warn("THREE.Color: Unknown color model "+e)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(e)){const r=s[1],o=r.length;if(o===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,t);if(o===6)return this.setHex(parseInt(r,16),t);console.warn("THREE.Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=Ot){const i=ed[e.toLowerCase()];return i!==void 0?this.setHex(i,t):console.warn("THREE.Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=$n(e.r),this.g=$n(e.g),this.b=$n(e.b),this}copyLinearToSRGB(e){return this.r=is(e.r),this.g=is(e.g),this.b=is(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=Ot){return tt.workingToColorSpace(Bt.copy(this),e),Math.round(Ze(Bt.r*255,0,255))*65536+Math.round(Ze(Bt.g*255,0,255))*256+Math.round(Ze(Bt.b*255,0,255))}getHexString(e=Ot){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=tt.workingColorSpace){tt.workingToColorSpace(Bt.copy(this),t);const i=Bt.r,s=Bt.g,r=Bt.b,o=Math.max(i,s,r),a=Math.min(i,s,r);let l,c;const u=(a+o)/2;if(a===o)l=0,c=0;else{const h=o-a;switch(c=u<=.5?h/(o+a):h/(2-o-a),o){case i:l=(s-r)/h+(s<r?6:0);break;case s:l=(r-i)/h+2;break;case r:l=(i-s)/h+4;break}l/=6}return e.h=l,e.s=c,e.l=u,e}getRGB(e,t=tt.workingColorSpace){return tt.workingToColorSpace(Bt.copy(this),t),e.r=Bt.r,e.g=Bt.g,e.b=Bt.b,e}getStyle(e=Ot){tt.workingToColorSpace(Bt.copy(this),e);const t=Bt.r,i=Bt.g,s=Bt.b;return e!==Ot?`color(${e} ${t.toFixed(3)} ${i.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(i*255)},${Math.round(s*255)})`}offsetHSL(e,t,i){return this.getHSL(Kn),this.setHSL(Kn.h+e,Kn.s+t,Kn.l+i)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,i){return this.r=e.r+(t.r-e.r)*i,this.g=e.g+(t.g-e.g)*i,this.b=e.b+(t.b-e.b)*i,this}lerpHSL(e,t){this.getHSL(Kn),e.getHSL(yr);const i=No(Kn.h,yr.h,t),s=No(Kn.s,yr.s,t),r=No(Kn.l,yr.l,t);return this.setHSL(i,s,r),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,i=this.g,s=this.b,r=e.elements;return this.r=r[0]*t+r[3]*i+r[6]*s,this.g=r[1]*t+r[4]*i+r[7]*s,this.b=r[2]*t+r[5]*i+r[8]*s,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const Bt=new Qe;Qe.NAMES=ed;let p_=0;class ps extends Ri{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:p_++}),this.uuid=Xn(),this.name="",this.type="Material",this.blending=ns,this.side=oi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=La,this.blendDst=Da,this.blendEquation=yi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Qe(0,0,0),this.blendAlpha=0,this.depthFunc=as,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Xl,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Di,this.stencilZFail=Di,this.stencilZPass=Di,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const i=e[t];if(i===void 0){console.warn(`THREE.Material: parameter '${t}' has value of undefined.`);continue}const s=this[t];if(s===void 0){console.warn(`THREE.Material: '${t}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(i):s&&s.isVector3&&i&&i.isVector3?s.copy(i):this[t]=i}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const i={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};i.uuid=this.uuid,i.type=this.type,this.name!==""&&(i.name=this.name),this.color&&this.color.isColor&&(i.color=this.color.getHex()),this.roughness!==void 0&&(i.roughness=this.roughness),this.metalness!==void 0&&(i.metalness=this.metalness),this.sheen!==void 0&&(i.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(i.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(i.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(i.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(i.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(i.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(i.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(i.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(i.shininess=this.shininess),this.clearcoat!==void 0&&(i.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(i.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(i.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(i.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(i.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,i.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(i.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(i.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(i.dispersion=this.dispersion),this.iridescence!==void 0&&(i.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(i.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(i.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(i.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(i.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(i.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(i.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(i.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(i.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(i.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(i.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(i.lightMap=this.lightMap.toJSON(e).uuid,i.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(i.aoMap=this.aoMap.toJSON(e).uuid,i.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(i.bumpMap=this.bumpMap.toJSON(e).uuid,i.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(i.normalMap=this.normalMap.toJSON(e).uuid,i.normalMapType=this.normalMapType,i.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(i.displacementMap=this.displacementMap.toJSON(e).uuid,i.displacementScale=this.displacementScale,i.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(i.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(i.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(i.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(i.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(i.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(i.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(i.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(i.combine=this.combine)),this.envMapRotation!==void 0&&(i.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(i.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(i.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(i.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(i.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(i.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(i.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(i.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(i.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(i.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(i.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(i.size=this.size),this.shadowSide!==null&&(i.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(i.sizeAttenuation=this.sizeAttenuation),this.blending!==ns&&(i.blending=this.blending),this.side!==oi&&(i.side=this.side),this.vertexColors===!0&&(i.vertexColors=!0),this.opacity<1&&(i.opacity=this.opacity),this.transparent===!0&&(i.transparent=!0),this.blendSrc!==La&&(i.blendSrc=this.blendSrc),this.blendDst!==Da&&(i.blendDst=this.blendDst),this.blendEquation!==yi&&(i.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(i.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(i.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(i.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(i.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(i.blendAlpha=this.blendAlpha),this.depthFunc!==as&&(i.depthFunc=this.depthFunc),this.depthTest===!1&&(i.depthTest=this.depthTest),this.depthWrite===!1&&(i.depthWrite=this.depthWrite),this.colorWrite===!1&&(i.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(i.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Xl&&(i.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(i.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(i.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==Di&&(i.stencilFail=this.stencilFail),this.stencilZFail!==Di&&(i.stencilZFail=this.stencilZFail),this.stencilZPass!==Di&&(i.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(i.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(i.rotation=this.rotation),this.polygonOffset===!0&&(i.polygonOffset=!0),this.polygonOffsetFactor!==0&&(i.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(i.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(i.linewidth=this.linewidth),this.dashSize!==void 0&&(i.dashSize=this.dashSize),this.gapSize!==void 0&&(i.gapSize=this.gapSize),this.scale!==void 0&&(i.scale=this.scale),this.dithering===!0&&(i.dithering=!0),this.alphaTest>0&&(i.alphaTest=this.alphaTest),this.alphaHash===!0&&(i.alphaHash=!0),this.alphaToCoverage===!0&&(i.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(i.premultipliedAlpha=!0),this.forceSinglePass===!0&&(i.forceSinglePass=!0),this.wireframe===!0&&(i.wireframe=!0),this.wireframeLinewidth>1&&(i.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(i.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(i.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(i.flatShading=!0),this.visible===!1&&(i.visible=!1),this.toneMapped===!1&&(i.toneMapped=!1),this.fog===!1&&(i.fog=!1),Object.keys(this.userData).length>0&&(i.userData=this.userData);function s(r){const o=[];for(const a in r){const l=r[a];delete l.metadata,o.push(l)}return o}if(t){const r=s(e.textures),o=s(e.images);r.length>0&&(i.textures=r),o.length>0&&(i.images=o)}return i}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let i=null;if(t!==null){const s=t.length;i=new Array(s);for(let r=0;r!==s;++r)i[r]=t[r].clone()}return this.clippingPlanes=i,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}class qt extends ps{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Qe(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Yn,this.combine=Hh,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Tt=new R,Sr=new ue;let m_=0;class hn{constructor(e,t,i=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:m_++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=i,this.usage=vc,this.updateRanges=[],this.gpuType=Wn,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,i){e*=this.itemSize,i*=t.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[e+s]=t.array[i+s];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,i=this.count;t<i;t++)Sr.fromBufferAttribute(this,t),Sr.applyMatrix3(e),this.setXY(t,Sr.x,Sr.y);else if(this.itemSize===3)for(let t=0,i=this.count;t<i;t++)Tt.fromBufferAttribute(this,t),Tt.applyMatrix3(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}applyMatrix4(e){for(let t=0,i=this.count;t<i;t++)Tt.fromBufferAttribute(this,t),Tt.applyMatrix4(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}applyNormalMatrix(e){for(let t=0,i=this.count;t<i;t++)Tt.fromBufferAttribute(this,t),Tt.applyNormalMatrix(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}transformDirection(e){for(let t=0,i=this.count;t<i;t++)Tt.fromBufferAttribute(this,t),Tt.transformDirection(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let i=this.array[e*this.itemSize+t];return this.normalized&&(i=wn(i,this.array)),i}setComponent(e,t,i){return this.normalized&&(i=ot(i,this.array)),this.array[e*this.itemSize+t]=i,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=wn(t,this.array)),t}setX(e,t){return this.normalized&&(t=ot(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=wn(t,this.array)),t}setY(e,t){return this.normalized&&(t=ot(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=wn(t,this.array)),t}setZ(e,t){return this.normalized&&(t=ot(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=wn(t,this.array)),t}setW(e,t){return this.normalized&&(t=ot(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,i){return e*=this.itemSize,this.normalized&&(t=ot(t,this.array),i=ot(i,this.array)),this.array[e+0]=t,this.array[e+1]=i,this}setXYZ(e,t,i,s){return e*=this.itemSize,this.normalized&&(t=ot(t,this.array),i=ot(i,this.array),s=ot(s,this.array)),this.array[e+0]=t,this.array[e+1]=i,this.array[e+2]=s,this}setXYZW(e,t,i,s,r){return e*=this.itemSize,this.normalized&&(t=ot(t,this.array),i=ot(i,this.array),s=ot(s,this.array),r=ot(r,this.array)),this.array[e+0]=t,this.array[e+1]=i,this.array[e+2]=s,this.array[e+3]=r,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==vc&&(e.usage=this.usage),e}}class td extends hn{constructor(e,t,i){super(new Uint16Array(e),t,i)}}class nd extends hn{constructor(e,t,i){super(new Uint32Array(e),t,i)}}class gt extends hn{constructor(e,t,i){super(new Float32Array(e),t,i)}}let g_=0;const an=new ft,Qo=new Lt,Gi=new R,nn=new Qs,Es=new Qs,Ut=new R;class it extends Ri{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:g_++}),this.uuid=Xn(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(Kh(e)?nd:td)(e,1):this.index=e,this}setIndirect(e){return this.indirect=e,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,i=0){this.groups.push({start:e,count:t,materialIndex:i})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const i=this.attributes.normal;if(i!==void 0){const r=new Ve().getNormalMatrix(e);i.applyNormalMatrix(r),i.needsUpdate=!0}const s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(e),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return an.makeRotationFromQuaternion(e),this.applyMatrix4(an),this}rotateX(e){return an.makeRotationX(e),this.applyMatrix4(an),this}rotateY(e){return an.makeRotationY(e),this.applyMatrix4(an),this}rotateZ(e){return an.makeRotationZ(e),this.applyMatrix4(an),this}translate(e,t,i){return an.makeTranslation(e,t,i),this.applyMatrix4(an),this}scale(e,t,i){return an.makeScale(e,t,i),this.applyMatrix4(an),this}lookAt(e){return Qo.lookAt(e),Qo.updateMatrix(),this.applyMatrix4(Qo.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Gi).negate(),this.translate(Gi.x,Gi.y,Gi.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const i=[];for(let s=0,r=e.length;s<r;s++){const o=e[s];i.push(o.x,o.y,o.z||0)}this.setAttribute("position",new gt(i,3))}else{const i=Math.min(e.length,t.count);for(let s=0;s<i;s++){const r=e[s];t.setXYZ(s,r.x,r.y,r.z||0)}e.length>t.count&&console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Qs);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new R(-1/0,-1/0,-1/0),new R(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let i=0,s=t.length;i<s;i++){const r=t[i];nn.setFromBufferAttribute(r),this.morphTargetsRelative?(Ut.addVectors(this.boundingBox.min,nn.min),this.boundingBox.expandByPoint(Ut),Ut.addVectors(this.boundingBox.max,nn.max),this.boundingBox.expandByPoint(Ut)):(this.boundingBox.expandByPoint(nn.min),this.boundingBox.expandByPoint(nn.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new xo);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new R,1/0);return}if(e){const i=this.boundingSphere.center;if(nn.setFromBufferAttribute(e),t)for(let r=0,o=t.length;r<o;r++){const a=t[r];Es.setFromBufferAttribute(a),this.morphTargetsRelative?(Ut.addVectors(nn.min,Es.min),nn.expandByPoint(Ut),Ut.addVectors(nn.max,Es.max),nn.expandByPoint(Ut)):(nn.expandByPoint(Es.min),nn.expandByPoint(Es.max))}nn.getCenter(i);let s=0;for(let r=0,o=e.count;r<o;r++)Ut.fromBufferAttribute(e,r),s=Math.max(s,i.distanceToSquared(Ut));if(t)for(let r=0,o=t.length;r<o;r++){const a=t[r],l=this.morphTargetsRelative;for(let c=0,u=a.count;c<u;c++)Ut.fromBufferAttribute(a,c),l&&(Gi.fromBufferAttribute(e,c),Ut.add(Gi)),s=Math.max(s,i.distanceToSquared(Ut))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const i=t.position,s=t.normal,r=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new hn(new Float32Array(4*i.count),4));const o=this.getAttribute("tangent"),a=[],l=[];for(let L=0;L<i.count;L++)a[L]=new R,l[L]=new R;const c=new R,u=new R,h=new R,d=new ue,f=new ue,g=new ue,_=new R,m=new R;function p(L,y,b){c.fromBufferAttribute(i,L),u.fromBufferAttribute(i,y),h.fromBufferAttribute(i,b),d.fromBufferAttribute(r,L),f.fromBufferAttribute(r,y),g.fromBufferAttribute(r,b),u.sub(c),h.sub(c),f.sub(d),g.sub(d);const N=1/(f.x*g.y-g.x*f.y);isFinite(N)&&(_.copy(u).multiplyScalar(g.y).addScaledVector(h,-f.y).multiplyScalar(N),m.copy(h).multiplyScalar(f.x).addScaledVector(u,-g.x).multiplyScalar(N),a[L].add(_),a[y].add(_),a[b].add(_),l[L].add(m),l[y].add(m),l[b].add(m))}let C=this.groups;C.length===0&&(C=[{start:0,count:e.count}]);for(let L=0,y=C.length;L<y;++L){const b=C[L],N=b.start,O=b.count;for(let G=N,V=N+O;G<V;G+=3)p(e.getX(G+0),e.getX(G+1),e.getX(G+2))}const M=new R,x=new R,P=new R,w=new R;function T(L){P.fromBufferAttribute(s,L),w.copy(P);const y=a[L];M.copy(y),M.sub(P.multiplyScalar(P.dot(y))).normalize(),x.crossVectors(w,y);const N=x.dot(l[L])<0?-1:1;o.setXYZW(L,M.x,M.y,M.z,N)}for(let L=0,y=C.length;L<y;++L){const b=C[L],N=b.start,O=b.count;for(let G=N,V=N+O;G<V;G+=3)T(e.getX(G+0)),T(e.getX(G+1)),T(e.getX(G+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let i=this.getAttribute("normal");if(i===void 0)i=new hn(new Float32Array(t.count*3),3),this.setAttribute("normal",i);else for(let d=0,f=i.count;d<f;d++)i.setXYZ(d,0,0,0);const s=new R,r=new R,o=new R,a=new R,l=new R,c=new R,u=new R,h=new R;if(e)for(let d=0,f=e.count;d<f;d+=3){const g=e.getX(d+0),_=e.getX(d+1),m=e.getX(d+2);s.fromBufferAttribute(t,g),r.fromBufferAttribute(t,_),o.fromBufferAttribute(t,m),u.subVectors(o,r),h.subVectors(s,r),u.cross(h),a.fromBufferAttribute(i,g),l.fromBufferAttribute(i,_),c.fromBufferAttribute(i,m),a.add(u),l.add(u),c.add(u),i.setXYZ(g,a.x,a.y,a.z),i.setXYZ(_,l.x,l.y,l.z),i.setXYZ(m,c.x,c.y,c.z)}else for(let d=0,f=t.count;d<f;d+=3)s.fromBufferAttribute(t,d+0),r.fromBufferAttribute(t,d+1),o.fromBufferAttribute(t,d+2),u.subVectors(o,r),h.subVectors(s,r),u.cross(h),i.setXYZ(d+0,u.x,u.y,u.z),i.setXYZ(d+1,u.x,u.y,u.z),i.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),i.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,i=e.count;t<i;t++)Ut.fromBufferAttribute(e,t),Ut.normalize(),e.setXYZ(t,Ut.x,Ut.y,Ut.z)}toNonIndexed(){function e(a,l){const c=a.array,u=a.itemSize,h=a.normalized,d=new c.constructor(l.length*u);let f=0,g=0;for(let _=0,m=l.length;_<m;_++){a.isInterleavedBufferAttribute?f=l[_]*a.data.stride+a.offset:f=l[_]*u;for(let p=0;p<u;p++)d[g++]=c[f++]}return new hn(d,u,h)}if(this.index===null)return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new it,i=this.index.array,s=this.attributes;for(const a in s){const l=s[a],c=e(l,i);t.setAttribute(a,c)}const r=this.morphAttributes;for(const a in r){const l=[],c=r[a];for(let u=0,h=c.length;u<h;u++){const d=c[u],f=e(d,i);l.push(f)}t.morphAttributes[a]=l}t.morphTargetsRelative=this.morphTargetsRelative;const o=this.groups;for(let a=0,l=o.length;a<l;a++){const c=o[a];t.addGroup(c.start,c.count,c.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const l=this.parameters;for(const c in l)l[c]!==void 0&&(e[c]=l[c]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const i=this.attributes;for(const l in i){const c=i[l];e.data.attributes[l]=c.toJSON(e.data)}const s={};let r=!1;for(const l in this.morphAttributes){const c=this.morphAttributes[l],u=[];for(let h=0,d=c.length;h<d;h++){const f=c[h];u.push(f.toJSON(e.data))}u.length>0&&(s[l]=u,r=!0)}r&&(e.data.morphAttributes=s,e.data.morphTargetsRelative=this.morphTargetsRelative);const o=this.groups;o.length>0&&(e.data.groups=JSON.parse(JSON.stringify(o)));const a=this.boundingSphere;return a!==null&&(e.data.boundingSphere=a.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const i=e.index;i!==null&&this.setIndex(i.clone());const s=e.attributes;for(const c in s){const u=s[c];this.setAttribute(c,u.clone(t))}const r=e.morphAttributes;for(const c in r){const u=[],h=r[c];for(let d=0,f=h.length;d<f;d++)u.push(h[d].clone(t));this.morphAttributes[c]=u}this.morphTargetsRelative=e.morphTargetsRelative;const o=e.groups;for(let c=0,u=o.length;c<u;c++){const h=o[c];this.addGroup(h.start,h.count,h.materialIndex)}const a=e.boundingBox;a!==null&&(this.boundingBox=a.clone());const l=e.boundingSphere;return l!==null&&(this.boundingSphere=l.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}const ru=new ft,fi=new yo,br=new xo,ou=new R,Mr=new R,wr=new R,Er=new R,ea=new R,Tr=new R,au=new R,Ar=new R;class xt extends Lt{constructor(e=new it,t=new qt){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){const s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}getVertexPosition(e,t){const i=this.geometry,s=i.attributes.position,r=i.morphAttributes.position,o=i.morphTargetsRelative;t.fromBufferAttribute(s,e);const a=this.morphTargetInfluences;if(r&&a){Tr.set(0,0,0);for(let l=0,c=r.length;l<c;l++){const u=a[l],h=r[l];u!==0&&(ea.fromBufferAttribute(h,e),o?Tr.addScaledVector(ea,u):Tr.addScaledVector(ea.sub(t),u))}t.add(Tr)}return t}raycast(e,t){const i=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(i.boundingSphere===null&&i.computeBoundingSphere(),br.copy(i.boundingSphere),br.applyMatrix4(r),fi.copy(e.ray).recast(e.near),!(br.containsPoint(fi.origin)===!1&&(fi.intersectSphere(br,ou)===null||fi.origin.distanceToSquared(ou)>(e.far-e.near)**2))&&(ru.copy(r).invert(),fi.copy(e.ray).applyMatrix4(ru),!(i.boundingBox!==null&&fi.intersectsBox(i.boundingBox)===!1)&&this._computeIntersections(e,t,fi)))}_computeIntersections(e,t,i){let s;const r=this.geometry,o=this.material,a=r.index,l=r.attributes.position,c=r.attributes.uv,u=r.attributes.uv1,h=r.attributes.normal,d=r.groups,f=r.drawRange;if(a!==null)if(Array.isArray(o))for(let g=0,_=d.length;g<_;g++){const m=d[g],p=o[m.materialIndex],C=Math.max(m.start,f.start),M=Math.min(a.count,Math.min(m.start+m.count,f.start+f.count));for(let x=C,P=M;x<P;x+=3){const w=a.getX(x),T=a.getX(x+1),L=a.getX(x+2);s=Cr(this,p,e,i,c,u,h,w,T,L),s&&(s.faceIndex=Math.floor(x/3),s.face.materialIndex=m.materialIndex,t.push(s))}}else{const g=Math.max(0,f.start),_=Math.min(a.count,f.start+f.count);for(let m=g,p=_;m<p;m+=3){const C=a.getX(m),M=a.getX(m+1),x=a.getX(m+2);s=Cr(this,o,e,i,c,u,h,C,M,x),s&&(s.faceIndex=Math.floor(m/3),t.push(s))}}else if(l!==void 0)if(Array.isArray(o))for(let g=0,_=d.length;g<_;g++){const m=d[g],p=o[m.materialIndex],C=Math.max(m.start,f.start),M=Math.min(l.count,Math.min(m.start+m.count,f.start+f.count));for(let x=C,P=M;x<P;x+=3){const w=x,T=x+1,L=x+2;s=Cr(this,p,e,i,c,u,h,w,T,L),s&&(s.faceIndex=Math.floor(x/3),s.face.materialIndex=m.materialIndex,t.push(s))}}else{const g=Math.max(0,f.start),_=Math.min(l.count,f.start+f.count);for(let m=g,p=_;m<p;m+=3){const C=m,M=m+1,x=m+2;s=Cr(this,o,e,i,c,u,h,C,M,x),s&&(s.faceIndex=Math.floor(m/3),t.push(s))}}}}function __(n,e,t,i,s,r,o,a){let l;if(e.side===Jt?l=i.intersectTriangle(o,r,s,!0,a):l=i.intersectTriangle(s,r,o,e.side===oi,a),l===null)return null;Ar.copy(a),Ar.applyMatrix4(n.matrixWorld);const c=t.ray.origin.distanceTo(Ar);return c<t.near||c>t.far?null:{distance:c,point:Ar.clone(),object:n}}function Cr(n,e,t,i,s,r,o,a,l,c){n.getVertexPosition(a,Mr),n.getVertexPosition(l,wr),n.getVertexPosition(c,Er);const u=__(n,e,t,i,Mr,wr,Er,au);if(u){const h=new R;ln.getBarycoord(au,Mr,wr,Er,h),s&&(u.uv=ln.getInterpolatedAttribute(s,a,l,c,h,new ue)),r&&(u.uv1=ln.getInterpolatedAttribute(r,a,l,c,h,new ue)),o&&(u.normal=ln.getInterpolatedAttribute(o,a,l,c,h,new R),u.normal.dot(i.direction)>0&&u.normal.multiplyScalar(-1));const d={a,b:l,c,normal:new R,materialIndex:0};ln.getNormal(Mr,wr,Er,d.normal),u.face=d,u.barycoord=h}return u}class er extends it{constructor(e=1,t=1,i=1,s=1,r=1,o=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:i,widthSegments:s,heightSegments:r,depthSegments:o};const a=this;s=Math.floor(s),r=Math.floor(r),o=Math.floor(o);const l=[],c=[],u=[],h=[];let d=0,f=0;g("z","y","x",-1,-1,i,t,e,o,r,0),g("z","y","x",1,-1,i,t,-e,o,r,1),g("x","z","y",1,1,e,i,t,s,o,2),g("x","z","y",1,-1,e,i,-t,s,o,3),g("x","y","z",1,-1,e,t,i,s,r,4),g("x","y","z",-1,-1,e,t,-i,s,r,5),this.setIndex(l),this.setAttribute("position",new gt(c,3)),this.setAttribute("normal",new gt(u,3)),this.setAttribute("uv",new gt(h,2));function g(_,m,p,C,M,x,P,w,T,L,y){const b=x/T,N=P/L,O=x/2,G=P/2,V=w/2,X=T+1,q=L+1;let J=0,z=0;const H=new R;for(let D=0;D<q;D++){const W=D*N-G;for(let Q=0;Q<X;Q++){const me=Q*b-O;H[_]=me*C,H[m]=W*M,H[p]=V,c.push(H.x,H.y,H.z),H[_]=0,H[m]=0,H[p]=w>0?1:-1,u.push(H.x,H.y,H.z),h.push(Q/T),h.push(1-D/L),J+=1}}for(let D=0;D<L;D++)for(let W=0;W<T;W++){const Q=d+W+X*D,me=d+W+X*(D+1),De=d+(W+1)+X*(D+1),Ie=d+(W+1)+X*D;l.push(Q,me,Ie),l.push(me,De,Ie),z+=6}a.addGroup(f,z,y),f+=z,d+=J}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new er(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}function hs(n){const e={};for(const t in n){e[t]={};for(const i in n[t]){const s=n[t][i];s&&(s.isColor||s.isMatrix3||s.isMatrix4||s.isVector2||s.isVector3||s.isVector4||s.isTexture||s.isQuaternion)?s.isRenderTargetTexture?(console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][i]=null):e[t][i]=s.clone():Array.isArray(s)?e[t][i]=s.slice():e[t][i]=s}}return e}function $t(n){const e={};for(let t=0;t<n.length;t++){const i=hs(n[t]);for(const s in i)e[s]=i[s]}return e}function v_(n){const e=[];for(let t=0;t<n.length;t++)e.push(n[t].clone());return e}function id(n){const e=n.getRenderTarget();return e===null?n.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:tt.workingColorSpace}const x_={clone:hs,merge:$t};var y_=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,S_=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class ai extends ps{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=y_,this.fragmentShader=S_,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=hs(e.uniforms),this.uniformsGroups=v_(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const s in this.uniforms){const o=this.uniforms[s].value;o&&o.isTexture?t.uniforms[s]={type:"t",value:o.toJSON(e).uuid}:o&&o.isColor?t.uniforms[s]={type:"c",value:o.getHex()}:o&&o.isVector2?t.uniforms[s]={type:"v2",value:o.toArray()}:o&&o.isVector3?t.uniforms[s]={type:"v3",value:o.toArray()}:o&&o.isVector4?t.uniforms[s]={type:"v4",value:o.toArray()}:o&&o.isMatrix3?t.uniforms[s]={type:"m3",value:o.toArray()}:o&&o.isMatrix4?t.uniforms[s]={type:"m4",value:o.toArray()}:t.uniforms[s]={value:o}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const i={};for(const s in this.extensions)this.extensions[s]===!0&&(i[s]=!0);return Object.keys(i).length>0&&(t.extensions=i),t}}class sd extends Lt{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new ft,this.projectionMatrix=new ft,this.projectionMatrixInverse=new ft,this.coordinateSystem=An,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorldInverse.copy(this.matrixWorld).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorldInverse.copy(this.matrixWorld).invert()}clone(){return new this.constructor().copy(this)}}const Qn=new R,cu=new ue,lu=new ue;class sn extends sd{constructor(e=50,t=1,i=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=i,this.far=s,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=xc*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(Kr*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return xc*2*Math.atan(Math.tan(Kr*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,i){Qn.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(Qn.x,Qn.y).multiplyScalar(-e/Qn.z),Qn.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),i.set(Qn.x,Qn.y).multiplyScalar(-e/Qn.z)}getViewSize(e,t){return this.getViewBounds(e,cu,lu),t.subVectors(lu,cu)}setViewOffset(e,t,i,s,r,o){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(Kr*.5*this.fov)/this.zoom,i=2*t,s=this.aspect*i,r=-.5*s;const o=this.view;if(this.view!==null&&this.view.enabled){const l=o.fullWidth,c=o.fullHeight;r+=o.offsetX*s/l,t-=o.offsetY*i/c,s*=o.width/l,i*=o.height/c}const a=this.filmOffset;a!==0&&(r+=e*a/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,t,t-i,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}const Vi=-90,Wi=1;class b_ extends Lt{constructor(e,t,i){super(),this.type="CubeCamera",this.renderTarget=i,this.coordinateSystem=null,this.activeMipmapLevel=0;const s=new sn(Vi,Wi,e,t);s.layers=this.layers,this.add(s);const r=new sn(Vi,Wi,e,t);r.layers=this.layers,this.add(r);const o=new sn(Vi,Wi,e,t);o.layers=this.layers,this.add(o);const a=new sn(Vi,Wi,e,t);a.layers=this.layers,this.add(a);const l=new sn(Vi,Wi,e,t);l.layers=this.layers,this.add(l);const c=new sn(Vi,Wi,e,t);c.layers=this.layers,this.add(c)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[i,s,r,o,a,l]=t;for(const c of t)this.remove(c);if(e===An)i.up.set(0,1,0),i.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),o.up.set(0,0,1),o.lookAt(0,-1,0),a.up.set(0,1,0),a.lookAt(0,0,1),l.up.set(0,1,0),l.lookAt(0,0,-1);else if(e===ao)i.up.set(0,-1,0),i.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),o.up.set(0,0,-1),o.lookAt(0,-1,0),a.up.set(0,-1,0),a.lookAt(0,0,1),l.up.set(0,-1,0),l.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const c of t)this.add(c),c.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:i,activeMipmapLevel:s}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[r,o,a,l,c,u]=this.children,h=e.getRenderTarget(),d=e.getActiveCubeFace(),f=e.getActiveMipmapLevel(),g=e.xr.enabled;e.xr.enabled=!1;const _=i.texture.generateMipmaps;i.texture.generateMipmaps=!1,e.setRenderTarget(i,0,s),e.render(t,r),e.setRenderTarget(i,1,s),e.render(t,o),e.setRenderTarget(i,2,s),e.render(t,a),e.setRenderTarget(i,3,s),e.render(t,l),e.setRenderTarget(i,4,s),e.render(t,c),i.texture.generateMipmaps=_,e.setRenderTarget(i,5,s),e.render(t,u),e.setRenderTarget(h,d,f),e.xr.enabled=g,i.texture.needsPMREMUpdate=!0}}class rd extends Vt{constructor(e=[],t=cs,i,s,r,o,a,l,c,u){super(e,t,i,s,r,o,a,l,c,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class M_ extends Ai{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const i={width:e,height:e,depth:1},s=[i,i,i,i,i,i];this.texture=new rd(s),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const i={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new er(5,5,5),r=new ai({name:"CubemapFromEquirect",uniforms:hs(i.uniforms),vertexShader:i.vertexShader,fragmentShader:i.fragmentShader,side:Jt,blending:si});r.uniforms.tEquirect.value=t;const o=new xt(s,r),a=t.minFilter;return t.minFilter===wi&&(t.minFilter=Tn),new b_(1,10,this).update(e,o),t.minFilter=a,o.geometry.dispose(),o.material.dispose(),this}clear(e,t=!0,i=!0,s=!0){const r=e.getRenderTarget();for(let o=0;o<6;o++)e.setRenderTarget(this,o),e.clear(t,i,s);e.setRenderTarget(r)}}class Ft extends Lt{constructor(){super(),this.isGroup=!0,this.type="Group"}}const w_={type:"move"};class ta{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Ft,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Ft,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new R,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new R),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Ft,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new R,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new R),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const i of e.hand.values())this._getHandJoint(t,i)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,i){let s=null,r=null,o=null;const a=this._targetRay,l=this._grip,c=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(c&&e.hand){o=!0;for(const _ of e.hand.values()){const m=t.getJointPose(_,i),p=this._getHandJoint(c,_);m!==null&&(p.matrix.fromArray(m.transform.matrix),p.matrix.decompose(p.position,p.rotation,p.scale),p.matrixWorldNeedsUpdate=!0,p.jointRadius=m.radius),p.visible=m!==null}const u=c.joints["index-finger-tip"],h=c.joints["thumb-tip"],d=u.position.distanceTo(h.position),f=.02,g=.005;c.inputState.pinching&&d>f+g?(c.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!c.inputState.pinching&&d<=f-g&&(c.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else l!==null&&e.gripSpace&&(r=t.getPose(e.gripSpace,i),r!==null&&(l.matrix.fromArray(r.transform.matrix),l.matrix.decompose(l.position,l.rotation,l.scale),l.matrixWorldNeedsUpdate=!0,r.linearVelocity?(l.hasLinearVelocity=!0,l.linearVelocity.copy(r.linearVelocity)):l.hasLinearVelocity=!1,r.angularVelocity?(l.hasAngularVelocity=!0,l.angularVelocity.copy(r.angularVelocity)):l.hasAngularVelocity=!1));a!==null&&(s=t.getPose(e.targetRaySpace,i),s===null&&r!==null&&(s=r),s!==null&&(a.matrix.fromArray(s.transform.matrix),a.matrix.decompose(a.position,a.rotation,a.scale),a.matrixWorldNeedsUpdate=!0,s.linearVelocity?(a.hasLinearVelocity=!0,a.linearVelocity.copy(s.linearVelocity)):a.hasLinearVelocity=!1,s.angularVelocity?(a.hasAngularVelocity=!0,a.angularVelocity.copy(s.angularVelocity)):a.hasAngularVelocity=!1,this.dispatchEvent(w_)))}return a!==null&&(a.visible=s!==null),l!==null&&(l.visible=r!==null),c!==null&&(c.visible=o!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const i=new Ft;i.matrixAutoUpdate=!1,i.visible=!1,e.joints[t.jointName]=i,e.add(i)}return e.joints[t.jointName]}}class od extends Lt{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Yn,this.environmentIntensity=1,this.environmentRotation=new Yn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}class E_{constructor(e,t){this.isInterleavedBuffer=!0,this.array=e,this.stride=t,this.count=e!==void 0?e.length/t:0,this.usage=vc,this.updateRanges=[],this.version=0,this.uuid=Xn()}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.array=new e.array.constructor(e.array),this.count=e.count,this.stride=e.stride,this.usage=e.usage,this}copyAt(e,t,i){e*=this.stride,i*=t.stride;for(let s=0,r=this.stride;s<r;s++)this.array[e+s]=t.array[i+s];return this}set(e,t=0){return this.array.set(e,t),this}clone(e){e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=Xn()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=this.array.slice(0).buffer);const t=new this.array.constructor(e.arrayBuffers[this.array.buffer._uuid]),i=new this.constructor(t,this.stride);return i.setUsage(this.usage),i}onUpload(e){return this.onUploadCallback=e,this}toJSON(e){return e.arrayBuffers===void 0&&(e.arrayBuffers={}),this.array.buffer._uuid===void 0&&(this.array.buffer._uuid=Xn()),e.arrayBuffers[this.array.buffer._uuid]===void 0&&(e.arrayBuffers[this.array.buffer._uuid]=Array.from(new Uint32Array(this.array.buffer))),{uuid:this.uuid,buffer:this.array.buffer._uuid,type:this.array.constructor.name,stride:this.stride}}}const Xt=new R;class co{constructor(e,t,i,s=!1){this.isInterleavedBufferAttribute=!0,this.name="",this.data=e,this.itemSize=t,this.offset=i,this.normalized=s}get count(){return this.data.count}get array(){return this.data.array}set needsUpdate(e){this.data.needsUpdate=e}applyMatrix4(e){for(let t=0,i=this.data.count;t<i;t++)Xt.fromBufferAttribute(this,t),Xt.applyMatrix4(e),this.setXYZ(t,Xt.x,Xt.y,Xt.z);return this}applyNormalMatrix(e){for(let t=0,i=this.count;t<i;t++)Xt.fromBufferAttribute(this,t),Xt.applyNormalMatrix(e),this.setXYZ(t,Xt.x,Xt.y,Xt.z);return this}transformDirection(e){for(let t=0,i=this.count;t<i;t++)Xt.fromBufferAttribute(this,t),Xt.transformDirection(e),this.setXYZ(t,Xt.x,Xt.y,Xt.z);return this}getComponent(e,t){let i=this.array[e*this.data.stride+this.offset+t];return this.normalized&&(i=wn(i,this.array)),i}setComponent(e,t,i){return this.normalized&&(i=ot(i,this.array)),this.data.array[e*this.data.stride+this.offset+t]=i,this}setX(e,t){return this.normalized&&(t=ot(t,this.array)),this.data.array[e*this.data.stride+this.offset]=t,this}setY(e,t){return this.normalized&&(t=ot(t,this.array)),this.data.array[e*this.data.stride+this.offset+1]=t,this}setZ(e,t){return this.normalized&&(t=ot(t,this.array)),this.data.array[e*this.data.stride+this.offset+2]=t,this}setW(e,t){return this.normalized&&(t=ot(t,this.array)),this.data.array[e*this.data.stride+this.offset+3]=t,this}getX(e){let t=this.data.array[e*this.data.stride+this.offset];return this.normalized&&(t=wn(t,this.array)),t}getY(e){let t=this.data.array[e*this.data.stride+this.offset+1];return this.normalized&&(t=wn(t,this.array)),t}getZ(e){let t=this.data.array[e*this.data.stride+this.offset+2];return this.normalized&&(t=wn(t,this.array)),t}getW(e){let t=this.data.array[e*this.data.stride+this.offset+3];return this.normalized&&(t=wn(t,this.array)),t}setXY(e,t,i){return e=e*this.data.stride+this.offset,this.normalized&&(t=ot(t,this.array),i=ot(i,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this}setXYZ(e,t,i,s){return e=e*this.data.stride+this.offset,this.normalized&&(t=ot(t,this.array),i=ot(i,this.array),s=ot(s,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this.data.array[e+2]=s,this}setXYZW(e,t,i,s,r){return e=e*this.data.stride+this.offset,this.normalized&&(t=ot(t,this.array),i=ot(i,this.array),s=ot(s,this.array),r=ot(r,this.array)),this.data.array[e+0]=t,this.data.array[e+1]=i,this.data.array[e+2]=s,this.data.array[e+3]=r,this}clone(e){if(e===void 0){console.log("THREE.InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");const t=[];for(let i=0;i<this.count;i++){const s=i*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return new hn(new this.array.constructor(t),this.itemSize,this.normalized)}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.clone(e)),new co(e.interleavedBuffers[this.data.uuid],this.itemSize,this.offset,this.normalized)}toJSON(e){if(e===void 0){console.log("THREE.InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");const t=[];for(let i=0;i<this.count;i++){const s=i*this.data.stride+this.offset;for(let r=0;r<this.itemSize;r++)t.push(this.data.array[s+r])}return{itemSize:this.itemSize,type:this.array.constructor.name,array:t,normalized:this.normalized}}else return e.interleavedBuffers===void 0&&(e.interleavedBuffers={}),e.interleavedBuffers[this.data.uuid]===void 0&&(e.interleavedBuffers[this.data.uuid]=this.data.toJSON(e)),{isInterleavedBufferAttribute:!0,itemSize:this.itemSize,data:this.data.uuid,offset:this.offset,normalized:this.normalized}}}class Gc extends ps{constructor(e){super(),this.isSpriteMaterial=!0,this.type="SpriteMaterial",this.color=new Qe(16777215),this.map=null,this.alphaMap=null,this.rotation=0,this.sizeAttenuation=!0,this.transparent=!0,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.alphaMap=e.alphaMap,this.rotation=e.rotation,this.sizeAttenuation=e.sizeAttenuation,this.fog=e.fog,this}}let Xi;const Ts=new R,$i=new R,Yi=new R,qi=new ue,As=new ue,ad=new ft,Rr=new R,Cs=new R,Pr=new R,uu=new ue,na=new ue,hu=new ue;class Vc extends Lt{constructor(e=new Gc){if(super(),this.isSprite=!0,this.type="Sprite",Xi===void 0){Xi=new it;const t=new Float32Array([-.5,-.5,0,0,0,.5,-.5,0,1,0,.5,.5,0,1,1,-.5,.5,0,0,1]),i=new E_(t,5);Xi.setIndex([0,1,2,0,2,3]),Xi.setAttribute("position",new co(i,3,0,!1)),Xi.setAttribute("uv",new co(i,2,3,!1))}this.geometry=Xi,this.material=e,this.center=new ue(.5,.5),this.count=1}raycast(e,t){e.camera===null&&console.error('THREE.Sprite: "Raycaster.camera" needs to be set in order to raycast against sprites.'),$i.setFromMatrixScale(this.matrixWorld),ad.copy(e.camera.matrixWorld),this.modelViewMatrix.multiplyMatrices(e.camera.matrixWorldInverse,this.matrixWorld),Yi.setFromMatrixPosition(this.modelViewMatrix),e.camera.isPerspectiveCamera&&this.material.sizeAttenuation===!1&&$i.multiplyScalar(-Yi.z);const i=this.material.rotation;let s,r;i!==0&&(r=Math.cos(i),s=Math.sin(i));const o=this.center;Lr(Rr.set(-.5,-.5,0),Yi,o,$i,s,r),Lr(Cs.set(.5,-.5,0),Yi,o,$i,s,r),Lr(Pr.set(.5,.5,0),Yi,o,$i,s,r),uu.set(0,0),na.set(1,0),hu.set(1,1);let a=e.ray.intersectTriangle(Rr,Cs,Pr,!1,Ts);if(a===null&&(Lr(Cs.set(-.5,.5,0),Yi,o,$i,s,r),na.set(0,1),a=e.ray.intersectTriangle(Rr,Pr,Cs,!1,Ts),a===null))return;const l=e.ray.origin.distanceTo(Ts);l<e.near||l>e.far||t.push({distance:l,point:Ts.clone(),uv:ln.getInterpolation(Ts,Rr,Cs,Pr,uu,na,hu,new ue),face:null,object:this})}copy(e,t){return super.copy(e,t),e.center!==void 0&&this.center.copy(e.center),this.material=e.material,this}}function Lr(n,e,t,i,s,r){qi.subVectors(n,t).addScalar(.5).multiply(i),s!==void 0?(As.x=r*qi.x-s*qi.y,As.y=s*qi.x+r*qi.y):As.copy(qi),n.copy(e),n.x+=As.x,n.y+=As.y,n.applyMatrix4(ad)}const ia=new R,T_=new R,A_=new Ve;class Gn{constructor(e=new R(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,i,s){return this.normal.set(e,t,i),this.constant=s,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,i){const s=ia.subVectors(i,t).cross(T_.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(s,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const i=e.delta(ia),s=this.normal.dot(i);if(s===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const r=-(e.start.dot(this.normal)+this.constant)/s;return r<0||r>1?null:t.copy(e.start).addScaledVector(i,r)}intersectsLine(e){const t=this.distanceToPoint(e.start),i=this.distanceToPoint(e.end);return t<0&&i>0||i<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const i=t||A_.getNormalMatrix(e),s=this.coplanarPoint(ia).applyMatrix4(e),r=this.normal.applyMatrix3(i).normalize();return this.constant=-s.dot(r),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const pi=new xo,C_=new ue(.5,.5),Dr=new R;class Wc{constructor(e=new Gn,t=new Gn,i=new Gn,s=new Gn,r=new Gn,o=new Gn){this.planes=[e,t,i,s,r,o]}set(e,t,i,s,r,o){const a=this.planes;return a[0].copy(e),a[1].copy(t),a[2].copy(i),a[3].copy(s),a[4].copy(r),a[5].copy(o),this}copy(e){const t=this.planes;for(let i=0;i<6;i++)t[i].copy(e.planes[i]);return this}setFromProjectionMatrix(e,t=An,i=!1){const s=this.planes,r=e.elements,o=r[0],a=r[1],l=r[2],c=r[3],u=r[4],h=r[5],d=r[6],f=r[7],g=r[8],_=r[9],m=r[10],p=r[11],C=r[12],M=r[13],x=r[14],P=r[15];if(s[0].setComponents(c-o,f-u,p-g,P-C).normalize(),s[1].setComponents(c+o,f+u,p+g,P+C).normalize(),s[2].setComponents(c+a,f+h,p+_,P+M).normalize(),s[3].setComponents(c-a,f-h,p-_,P-M).normalize(),i)s[4].setComponents(l,d,m,x).normalize(),s[5].setComponents(c-l,f-d,p-m,P-x).normalize();else if(s[4].setComponents(c-l,f-d,p-m,P-x).normalize(),t===An)s[5].setComponents(c+l,f+d,p+m,P+x).normalize();else if(t===ao)s[5].setComponents(l,d,m,x).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),pi.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),pi.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(pi)}intersectsSprite(e){pi.center.set(0,0,0);const t=C_.distanceTo(e.center);return pi.radius=.7071067811865476+t,pi.applyMatrix4(e.matrixWorld),this.intersectsSphere(pi)}intersectsSphere(e){const t=this.planes,i=e.center,s=-e.radius;for(let r=0;r<6;r++)if(t[r].distanceToPoint(i)<s)return!1;return!0}intersectsBox(e){const t=this.planes;for(let i=0;i<6;i++){const s=t[i];if(Dr.x=s.normal.x>0?e.max.x:e.min.x,Dr.y=s.normal.y>0?e.max.y:e.min.y,Dr.z=s.normal.z>0?e.max.z:e.min.z,s.distanceToPoint(Dr)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let i=0;i<6;i++)if(t[i].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class Pn extends ps{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new Qe(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}}const lo=new R,uo=new R,du=new ft,Rs=new yo,Ir=new xo,sa=new R,fu=new R;class Ln extends Lt{constructor(e=new it,t=new Pn){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){const e=this.geometry;if(e.index===null){const t=e.attributes.position,i=[0];for(let s=1,r=t.count;s<r;s++)lo.fromBufferAttribute(t,s-1),uo.fromBufferAttribute(t,s),i[s]=i[s-1],i[s]+=lo.distanceTo(uo);e.setAttribute("lineDistance",new gt(i,1))}else console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,t){const i=this.geometry,s=this.matrixWorld,r=e.params.Line.threshold,o=i.drawRange;if(i.boundingSphere===null&&i.computeBoundingSphere(),Ir.copy(i.boundingSphere),Ir.applyMatrix4(s),Ir.radius+=r,e.ray.intersectsSphere(Ir)===!1)return;du.copy(s).invert(),Rs.copy(e.ray).applyMatrix4(du);const a=r/((this.scale.x+this.scale.y+this.scale.z)/3),l=a*a,c=this.isLineSegments?2:1,u=i.index,d=i.attributes.position;if(u!==null){const f=Math.max(0,o.start),g=Math.min(u.count,o.start+o.count);for(let _=f,m=g-1;_<m;_+=c){const p=u.getX(_),C=u.getX(_+1),M=Nr(this,e,Rs,l,p,C,_);M&&t.push(M)}if(this.isLineLoop){const _=u.getX(g-1),m=u.getX(f),p=Nr(this,e,Rs,l,_,m,g-1);p&&t.push(p)}}else{const f=Math.max(0,o.start),g=Math.min(d.count,o.start+o.count);for(let _=f,m=g-1;_<m;_+=c){const p=Nr(this,e,Rs,l,_,_+1,_);p&&t.push(p)}if(this.isLineLoop){const _=Nr(this,e,Rs,l,g-1,f,g-1);_&&t.push(_)}}}updateMorphTargets(){const t=this.geometry.morphAttributes,i=Object.keys(t);if(i.length>0){const s=t[i[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,o=s.length;r<o;r++){const a=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[a]=r}}}}}function Nr(n,e,t,i,s,r,o){const a=n.geometry.attributes.position;if(lo.fromBufferAttribute(a,s),uo.fromBufferAttribute(a,r),t.distanceSqToSegment(lo,uo,sa,fu)>i)return;sa.applyMatrix4(n.matrixWorld);const c=e.ray.origin.distanceTo(sa);if(!(c<e.near||c>e.far))return{distance:c,point:fu.clone().applyMatrix4(n.matrixWorld),index:o,face:null,faceIndex:null,barycoord:null,object:n}}const pu=new R,mu=new R;class ho extends Ln{constructor(e,t){super(e,t),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){const e=this.geometry;if(e.index===null){const t=e.attributes.position,i=[];for(let s=0,r=t.count;s<r;s+=2)pu.fromBufferAttribute(t,s),mu.fromBufferAttribute(t,s+1),i[s]=s===0?0:i[s-1],i[s+1]=i[s]+pu.distanceTo(mu);e.setAttribute("lineDistance",new gt(i,1))}else console.warn("THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}}class cd extends Ln{constructor(e,t){super(e,t),this.isLineLoop=!0,this.type="LineLoop"}}class tr extends Vt{constructor(e,t,i,s,r,o,a,l,c){super(e,t,i,s,r,o,a,l,c),this.isCanvasTexture=!0,this.needsUpdate=!0}}class ld extends Vt{constructor(e,t,i=Ei,s,r,o,a=xn,l=xn,c,u=Gs,h=1){if(u!==Gs&&u!==Vs)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const d={width:e,height:t,depth:h};super(d,s,r,o,a,l,u,i,c),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new Bc(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class ud extends Vt{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class $s extends it{constructor(e=1,t=32,i=0,s=Math.PI*2){super(),this.type="CircleGeometry",this.parameters={radius:e,segments:t,thetaStart:i,thetaLength:s},t=Math.max(3,t);const r=[],o=[],a=[],l=[],c=new R,u=new ue;o.push(0,0,0),a.push(0,0,1),l.push(.5,.5);for(let h=0,d=3;h<=t;h++,d+=3){const f=i+h/t*s;c.x=e*Math.cos(f),c.y=e*Math.sin(f),o.push(c.x,c.y,c.z),a.push(0,0,1),u.x=(o[d]/e+1)/2,u.y=(o[d+1]/e+1)/2,l.push(u.x,u.y)}for(let h=1;h<=t;h++)r.push(h,h+1,0);this.setIndex(r),this.setAttribute("position",new gt(o,3)),this.setAttribute("normal",new gt(a,3)),this.setAttribute("uv",new gt(l,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new $s(e.radius,e.segments,e.thetaStart,e.thetaLength)}}class So extends it{constructor(e=1,t=1,i=1,s=32,r=1,o=!1,a=0,l=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:i,radialSegments:s,heightSegments:r,openEnded:o,thetaStart:a,thetaLength:l};const c=this;s=Math.floor(s),r=Math.floor(r);const u=[],h=[],d=[],f=[];let g=0;const _=[],m=i/2;let p=0;C(),o===!1&&(e>0&&M(!0),t>0&&M(!1)),this.setIndex(u),this.setAttribute("position",new gt(h,3)),this.setAttribute("normal",new gt(d,3)),this.setAttribute("uv",new gt(f,2));function C(){const x=new R,P=new R;let w=0;const T=(t-e)/i;for(let L=0;L<=r;L++){const y=[],b=L/r,N=b*(t-e)+e;for(let O=0;O<=s;O++){const G=O/s,V=G*l+a,X=Math.sin(V),q=Math.cos(V);P.x=N*X,P.y=-b*i+m,P.z=N*q,h.push(P.x,P.y,P.z),x.set(X,T,q).normalize(),d.push(x.x,x.y,x.z),f.push(G,1-b),y.push(g++)}_.push(y)}for(let L=0;L<s;L++)for(let y=0;y<r;y++){const b=_[y][L],N=_[y+1][L],O=_[y+1][L+1],G=_[y][L+1];(e>0||y!==0)&&(u.push(b,N,G),w+=3),(t>0||y!==r-1)&&(u.push(N,O,G),w+=3)}c.addGroup(p,w,0),p+=w}function M(x){const P=g,w=new ue,T=new R;let L=0;const y=x===!0?e:t,b=x===!0?1:-1;for(let O=1;O<=s;O++)h.push(0,m*b,0),d.push(0,b,0),f.push(.5,.5),g++;const N=g;for(let O=0;O<=s;O++){const V=O/s*l+a,X=Math.cos(V),q=Math.sin(V);T.x=y*q,T.y=m*b,T.z=y*X,h.push(T.x,T.y,T.z),d.push(0,b,0),w.x=X*.5+.5,w.y=q*.5*b+.5,f.push(w.x,w.y),g++}for(let O=0;O<s;O++){const G=P+O,V=N+O;x===!0?u.push(V,V+1,G):u.push(V+1,V,G),L+=3}c.addGroup(p,L,x===!0?1:2),p+=L}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new So(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class bo extends So{constructor(e=1,t=1,i=32,s=1,r=!1,o=0,a=Math.PI*2){super(0,e,t,i,s,r,o,a),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:i,heightSegments:s,openEnded:r,thetaStart:o,thetaLength:a}}static fromJSON(e){return new bo(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class In{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){console.warn("THREE.Curve: .getPoint() not implemented.")}getPointAt(e,t){const i=this.getUtoTmapping(e);return this.getPoint(i,t)}getPoints(e=5){const t=[];for(let i=0;i<=e;i++)t.push(this.getPoint(i/e));return t}getSpacedPoints(e=5){const t=[];for(let i=0;i<=e;i++)t.push(this.getPointAt(i/e));return t}getLength(){const e=this.getLengths();return e[e.length-1]}getLengths(e=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===e+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;const t=[];let i,s=this.getPoint(0),r=0;t.push(0);for(let o=1;o<=e;o++)i=this.getPoint(o/e),r+=i.distanceTo(s),t.push(r),s=i;return this.cacheArcLengths=t,t}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(e,t=null){const i=this.getLengths();let s=0;const r=i.length;let o;t?o=t:o=e*i[r-1];let a=0,l=r-1,c;for(;a<=l;)if(s=Math.floor(a+(l-a)/2),c=i[s]-o,c<0)a=s+1;else if(c>0)l=s-1;else{l=s;break}if(s=l,i[s]===o)return s/(r-1);const u=i[s],d=i[s+1]-u,f=(o-u)/d;return(s+f)/(r-1)}getTangent(e,t){let s=e-1e-4,r=e+1e-4;s<0&&(s=0),r>1&&(r=1);const o=this.getPoint(s),a=this.getPoint(r),l=t||(o.isVector2?new ue:new R);return l.copy(a).sub(o).normalize(),l}getTangentAt(e,t){const i=this.getUtoTmapping(e);return this.getTangent(i,t)}computeFrenetFrames(e,t=!1){const i=new R,s=[],r=[],o=[],a=new R,l=new ft;for(let f=0;f<=e;f++){const g=f/e;s[f]=this.getTangentAt(g,new R)}r[0]=new R,o[0]=new R;let c=Number.MAX_VALUE;const u=Math.abs(s[0].x),h=Math.abs(s[0].y),d=Math.abs(s[0].z);u<=c&&(c=u,i.set(1,0,0)),h<=c&&(c=h,i.set(0,1,0)),d<=c&&i.set(0,0,1),a.crossVectors(s[0],i).normalize(),r[0].crossVectors(s[0],a),o[0].crossVectors(s[0],r[0]);for(let f=1;f<=e;f++){if(r[f]=r[f-1].clone(),o[f]=o[f-1].clone(),a.crossVectors(s[f-1],s[f]),a.length()>Number.EPSILON){a.normalize();const g=Math.acos(Ze(s[f-1].dot(s[f]),-1,1));r[f].applyMatrix4(l.makeRotationAxis(a,g))}o[f].crossVectors(s[f],r[f])}if(t===!0){let f=Math.acos(Ze(r[0].dot(r[e]),-1,1));f/=e,s[0].dot(a.crossVectors(r[0],r[e]))>0&&(f=-f);for(let g=1;g<=e;g++)r[g].applyMatrix4(l.makeRotationAxis(s[g],f*g)),o[g].crossVectors(s[g],r[g])}return{tangents:s,normals:r,binormals:o}}clone(){return new this.constructor().copy(this)}copy(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}toJSON(){const e={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return e.arcLengthDivisions=this.arcLengthDivisions,e.type=this.type,e}fromJSON(e){return this.arcLengthDivisions=e.arcLengthDivisions,this}}class Xc extends In{constructor(e=0,t=0,i=1,s=1,r=0,o=Math.PI*2,a=!1,l=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=e,this.aY=t,this.xRadius=i,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=o,this.aClockwise=a,this.aRotation=l}getPoint(e,t=new ue){const i=t,s=Math.PI*2;let r=this.aEndAngle-this.aStartAngle;const o=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(o?r=0:r=s),this.aClockwise===!0&&!o&&(r===s?r=-s:r=r-s);const a=this.aStartAngle+e*r;let l=this.aX+this.xRadius*Math.cos(a),c=this.aY+this.yRadius*Math.sin(a);if(this.aRotation!==0){const u=Math.cos(this.aRotation),h=Math.sin(this.aRotation),d=l-this.aX,f=c-this.aY;l=d*u-f*h+this.aX,c=d*h+f*u+this.aY}return i.set(l,c)}copy(e){return super.copy(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}toJSON(){const e=super.toJSON();return e.aX=this.aX,e.aY=this.aY,e.xRadius=this.xRadius,e.yRadius=this.yRadius,e.aStartAngle=this.aStartAngle,e.aEndAngle=this.aEndAngle,e.aClockwise=this.aClockwise,e.aRotation=this.aRotation,e}fromJSON(e){return super.fromJSON(e),this.aX=e.aX,this.aY=e.aY,this.xRadius=e.xRadius,this.yRadius=e.yRadius,this.aStartAngle=e.aStartAngle,this.aEndAngle=e.aEndAngle,this.aClockwise=e.aClockwise,this.aRotation=e.aRotation,this}}class R_ extends Xc{constructor(e,t,i,s,r,o){super(e,t,i,i,s,r,o),this.isArcCurve=!0,this.type="ArcCurve"}}function $c(){let n=0,e=0,t=0,i=0;function s(r,o,a,l){n=r,e=a,t=-3*r+3*o-2*a-l,i=2*r-2*o+a+l}return{initCatmullRom:function(r,o,a,l,c){s(o,a,c*(a-r),c*(l-o))},initNonuniformCatmullRom:function(r,o,a,l,c,u,h){let d=(o-r)/c-(a-r)/(c+u)+(a-o)/u,f=(a-o)/u-(l-o)/(u+h)+(l-a)/h;d*=u,f*=u,s(o,a,d,f)},calc:function(r){const o=r*r,a=o*r;return n+e*r+t*o+i*a}}}const Ur=new R,ra=new $c,oa=new $c,aa=new $c;class P_ extends In{constructor(e=[],t=!1,i="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=e,this.closed=t,this.curveType=i,this.tension=s}getPoint(e,t=new R){const i=t,s=this.points,r=s.length,o=(r-(this.closed?0:1))*e;let a=Math.floor(o),l=o-a;this.closed?a+=a>0?0:(Math.floor(Math.abs(a)/r)+1)*r:l===0&&a===r-1&&(a=r-2,l=1);let c,u;this.closed||a>0?c=s[(a-1)%r]:(Ur.subVectors(s[0],s[1]).add(s[0]),c=Ur);const h=s[a%r],d=s[(a+1)%r];if(this.closed||a+2<r?u=s[(a+2)%r]:(Ur.subVectors(s[r-1],s[r-2]).add(s[r-1]),u=Ur),this.curveType==="centripetal"||this.curveType==="chordal"){const f=this.curveType==="chordal"?.5:.25;let g=Math.pow(c.distanceToSquared(h),f),_=Math.pow(h.distanceToSquared(d),f),m=Math.pow(d.distanceToSquared(u),f);_<1e-4&&(_=1),g<1e-4&&(g=_),m<1e-4&&(m=_),ra.initNonuniformCatmullRom(c.x,h.x,d.x,u.x,g,_,m),oa.initNonuniformCatmullRom(c.y,h.y,d.y,u.y,g,_,m),aa.initNonuniformCatmullRom(c.z,h.z,d.z,u.z,g,_,m)}else this.curveType==="catmullrom"&&(ra.initCatmullRom(c.x,h.x,d.x,u.x,this.tension),oa.initCatmullRom(c.y,h.y,d.y,u.y,this.tension),aa.initCatmullRom(c.z,h.z,d.z,u.z,this.tension));return i.set(ra.calc(l),oa.calc(l),aa.calc(l)),i}copy(e){super.copy(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){const s=e.points[t];this.points.push(s.clone())}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}toJSON(){const e=super.toJSON();e.points=[];for(let t=0,i=this.points.length;t<i;t++){const s=this.points[t];e.points.push(s.toArray())}return e.closed=this.closed,e.curveType=this.curveType,e.tension=this.tension,e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){const s=e.points[t];this.points.push(new R().fromArray(s))}return this.closed=e.closed,this.curveType=e.curveType,this.tension=e.tension,this}}function gu(n,e,t,i,s){const r=(i-e)*.5,o=(s-t)*.5,a=n*n,l=n*a;return(2*t-2*i+r+o)*l+(-3*t+3*i-2*r-o)*a+r*n+t}function L_(n,e){const t=1-n;return t*t*e}function D_(n,e){return 2*(1-n)*n*e}function I_(n,e){return n*n*e}function Us(n,e,t,i){return L_(n,e)+D_(n,t)+I_(n,i)}function N_(n,e){const t=1-n;return t*t*t*e}function U_(n,e){const t=1-n;return 3*t*t*n*e}function O_(n,e){return 3*(1-n)*n*n*e}function F_(n,e){return n*n*n*e}function Os(n,e,t,i,s){return N_(n,e)+U_(n,t)+O_(n,i)+F_(n,s)}class hd extends In{constructor(e=new ue,t=new ue,i=new ue,s=new ue){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=e,this.v1=t,this.v2=i,this.v3=s}getPoint(e,t=new ue){const i=t,s=this.v0,r=this.v1,o=this.v2,a=this.v3;return i.set(Os(e,s.x,r.x,o.x,a.x),Os(e,s.y,r.y,o.y,a.y)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){const e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}}class k_ extends In{constructor(e=new R,t=new R,i=new R,s=new R){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=e,this.v1=t,this.v2=i,this.v3=s}getPoint(e,t=new R){const i=t,s=this.v0,r=this.v1,o=this.v2,a=this.v3;return i.set(Os(e,s.x,r.x,o.x,a.x),Os(e,s.y,r.y,o.y,a.y),Os(e,s.z,r.z,o.z,a.z)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this.v3.copy(e.v3),this}toJSON(){const e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e.v3=this.v3.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this.v3.fromArray(e.v3),this}}class dd extends In{constructor(e=new ue,t=new ue){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=e,this.v2=t}getPoint(e,t=new ue){const i=t;return e===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(e).add(this.v1)),i}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new ue){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){const e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}}class z_ extends In{constructor(e=new R,t=new R){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=e,this.v2=t}getPoint(e,t=new R){const i=t;return e===1?i.copy(this.v2):(i.copy(this.v2).sub(this.v1),i.multiplyScalar(e).add(this.v1)),i}getPointAt(e,t){return this.getPoint(e,t)}getTangent(e,t=new R){return t.subVectors(this.v2,this.v1).normalize()}getTangentAt(e,t){return this.getTangent(e,t)}copy(e){return super.copy(e),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){const e=super.toJSON();return e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}}class fd extends In{constructor(e=new ue,t=new ue,i=new ue){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=e,this.v1=t,this.v2=i}getPoint(e,t=new ue){const i=t,s=this.v0,r=this.v1,o=this.v2;return i.set(Us(e,s.x,r.x,o.x),Us(e,s.y,r.y,o.y)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){const e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}}class B_ extends In{constructor(e=new R,t=new R,i=new R){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=e,this.v1=t,this.v2=i}getPoint(e,t=new R){const i=t,s=this.v0,r=this.v1,o=this.v2;return i.set(Us(e,s.x,r.x,o.x),Us(e,s.y,r.y,o.y),Us(e,s.z,r.z,o.z)),i}copy(e){return super.copy(e),this.v0.copy(e.v0),this.v1.copy(e.v1),this.v2.copy(e.v2),this}toJSON(){const e=super.toJSON();return e.v0=this.v0.toArray(),e.v1=this.v1.toArray(),e.v2=this.v2.toArray(),e}fromJSON(e){return super.fromJSON(e),this.v0.fromArray(e.v0),this.v1.fromArray(e.v1),this.v2.fromArray(e.v2),this}}class pd extends In{constructor(e=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=e}getPoint(e,t=new ue){const i=t,s=this.points,r=(s.length-1)*e,o=Math.floor(r),a=r-o,l=s[o===0?o:o-1],c=s[o],u=s[o>s.length-2?s.length-1:o+1],h=s[o>s.length-3?s.length-1:o+2];return i.set(gu(a,l.x,c.x,u.x,h.x),gu(a,l.y,c.y,u.y,h.y)),i}copy(e){super.copy(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){const s=e.points[t];this.points.push(s.clone())}return this}toJSON(){const e=super.toJSON();e.points=[];for(let t=0,i=this.points.length;t<i;t++){const s=this.points[t];e.points.push(s.toArray())}return e}fromJSON(e){super.fromJSON(e),this.points=[];for(let t=0,i=e.points.length;t<i;t++){const s=e.points[t];this.points.push(new ue().fromArray(s))}return this}}var _u=Object.freeze({__proto__:null,ArcCurve:R_,CatmullRomCurve3:P_,CubicBezierCurve:hd,CubicBezierCurve3:k_,EllipseCurve:Xc,LineCurve:dd,LineCurve3:z_,QuadraticBezierCurve:fd,QuadraticBezierCurve3:B_,SplineCurve:pd});class H_ extends In{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1}add(e){this.curves.push(e)}closePath(){const e=this.curves[0].getPoint(0),t=this.curves[this.curves.length-1].getPoint(1);if(!e.equals(t)){const i=e.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new _u[i](t,e))}return this}getPoint(e,t){const i=e*this.getLength(),s=this.getCurveLengths();let r=0;for(;r<s.length;){if(s[r]>=i){const o=s[r]-i,a=this.curves[r],l=a.getLength(),c=l===0?0:1-o/l;return a.getPointAt(c,t)}r++}return null}getLength(){const e=this.getCurveLengths();return e[e.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;const e=[];let t=0;for(let i=0,s=this.curves.length;i<s;i++)t+=this.curves[i].getLength(),e.push(t);return this.cacheLengths=e,e}getSpacedPoints(e=40){const t=[];for(let i=0;i<=e;i++)t.push(this.getPoint(i/e));return this.autoClose&&t.push(t[0]),t}getPoints(e=12){const t=[];let i;for(let s=0,r=this.curves;s<r.length;s++){const o=r[s],a=o.isEllipseCurve?e*2:o.isLineCurve||o.isLineCurve3?1:o.isSplineCurve?e*o.points.length:e,l=o.getPoints(a);for(let c=0;c<l.length;c++){const u=l[c];i&&i.equals(u)||(t.push(u),i=u)}}return this.autoClose&&t.length>1&&!t[t.length-1].equals(t[0])&&t.push(t[0]),t}copy(e){super.copy(e),this.curves=[];for(let t=0,i=e.curves.length;t<i;t++){const s=e.curves[t];this.curves.push(s.clone())}return this.autoClose=e.autoClose,this}toJSON(){const e=super.toJSON();e.autoClose=this.autoClose,e.curves=[];for(let t=0,i=this.curves.length;t<i;t++){const s=this.curves[t];e.curves.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.autoClose=e.autoClose,this.curves=[];for(let t=0,i=e.curves.length;t<i;t++){const s=e.curves[t];this.curves.push(new _u[s.type]().fromJSON(s))}return this}}class vu extends H_{constructor(e){super(),this.type="Path",this.currentPoint=new ue,e&&this.setFromPoints(e)}setFromPoints(e){this.moveTo(e[0].x,e[0].y);for(let t=1,i=e.length;t<i;t++)this.lineTo(e[t].x,e[t].y);return this}moveTo(e,t){return this.currentPoint.set(e,t),this}lineTo(e,t){const i=new dd(this.currentPoint.clone(),new ue(e,t));return this.curves.push(i),this.currentPoint.set(e,t),this}quadraticCurveTo(e,t,i,s){const r=new fd(this.currentPoint.clone(),new ue(e,t),new ue(i,s));return this.curves.push(r),this.currentPoint.set(i,s),this}bezierCurveTo(e,t,i,s,r,o){const a=new hd(this.currentPoint.clone(),new ue(e,t),new ue(i,s),new ue(r,o));return this.curves.push(a),this.currentPoint.set(r,o),this}splineThru(e){const t=[this.currentPoint.clone()].concat(e),i=new pd(t);return this.curves.push(i),this.currentPoint.copy(e[e.length-1]),this}arc(e,t,i,s,r,o){const a=this.currentPoint.x,l=this.currentPoint.y;return this.absarc(e+a,t+l,i,s,r,o),this}absarc(e,t,i,s,r,o){return this.absellipse(e,t,i,i,s,r,o),this}ellipse(e,t,i,s,r,o,a,l){const c=this.currentPoint.x,u=this.currentPoint.y;return this.absellipse(e+c,t+u,i,s,r,o,a,l),this}absellipse(e,t,i,s,r,o,a,l){const c=new Xc(e,t,i,s,r,o,a,l);if(this.curves.length>0){const h=c.getPoint(0);h.equals(this.currentPoint)||this.lineTo(h.x,h.y)}this.curves.push(c);const u=c.getPoint(1);return this.currentPoint.copy(u),this}copy(e){return super.copy(e),this.currentPoint.copy(e.currentPoint),this}toJSON(){const e=super.toJSON();return e.currentPoint=this.currentPoint.toArray(),e}fromJSON(e){return super.fromJSON(e),this.currentPoint.fromArray(e.currentPoint),this}}class Mo extends vu{constructor(e){super(e),this.uuid=Xn(),this.type="Shape",this.holes=[]}getPointsHoles(e){const t=[];for(let i=0,s=this.holes.length;i<s;i++)t[i]=this.holes[i].getPoints(e);return t}extractPoints(e){return{shape:this.getPoints(e),holes:this.getPointsHoles(e)}}copy(e){super.copy(e),this.holes=[];for(let t=0,i=e.holes.length;t<i;t++){const s=e.holes[t];this.holes.push(s.clone())}return this}toJSON(){const e=super.toJSON();e.uuid=this.uuid,e.holes=[];for(let t=0,i=this.holes.length;t<i;t++){const s=this.holes[t];e.holes.push(s.toJSON())}return e}fromJSON(e){super.fromJSON(e),this.uuid=e.uuid,this.holes=[];for(let t=0,i=e.holes.length;t<i;t++){const s=e.holes[t];this.holes.push(new vu().fromJSON(s))}return this}}function G_(n,e,t=2){const i=e&&e.length,s=i?e[0]*t:n.length;let r=md(n,0,s,t,!0);const o=[];if(!r||r.next===r.prev)return o;let a,l,c;if(i&&(r=Y_(n,e,r,t)),n.length>80*t){a=1/0,l=1/0;let u=-1/0,h=-1/0;for(let d=t;d<s;d+=t){const f=n[d],g=n[d+1];f<a&&(a=f),g<l&&(l=g),f>u&&(u=f),g>h&&(h=g)}c=Math.max(u-a,h-l),c=c!==0?32767/c:0}return Ys(r,o,t,a,l,c,0),o}function md(n,e,t,i,s){let r;if(s===sv(n,e,t,i)>0)for(let o=e;o<t;o+=i)r=xu(o/i|0,n[o],n[o+1],r);else for(let o=t-i;o>=e;o-=i)r=xu(o/i|0,n[o],n[o+1],r);return r&&ds(r,r.next)&&(js(r),r=r.next),r}function Ci(n,e){if(!n)return n;e||(e=n);let t=n,i;do if(i=!1,!t.steiner&&(ds(t,t.next)||bt(t.prev,t,t.next)===0)){if(js(t),t=e=t.prev,t===t.next)break;i=!0}else t=t.next;while(i||t!==e);return e}function Ys(n,e,t,i,s,r,o){if(!n)return;!o&&r&&K_(n,i,s,r);let a=n;for(;n.prev!==n.next;){const l=n.prev,c=n.next;if(r?W_(n,i,s,r):V_(n)){e.push(l.i,n.i,c.i),js(n),n=c.next,a=c.next;continue}if(n=c,n===a){o?o===1?(n=X_(Ci(n),e),Ys(n,e,t,i,s,r,2)):o===2&&$_(n,e,t,i,s,r):Ys(Ci(n),e,t,i,s,r,1);break}}}function V_(n){const e=n.prev,t=n,i=n.next;if(bt(e,t,i)>=0)return!1;const s=e.x,r=t.x,o=i.x,a=e.y,l=t.y,c=i.y,u=Math.min(s,r,o),h=Math.min(a,l,c),d=Math.max(s,r,o),f=Math.max(a,l,c);let g=i.next;for(;g!==e;){if(g.x>=u&&g.x<=d&&g.y>=h&&g.y<=f&&Ls(s,a,r,l,o,c,g.x,g.y)&&bt(g.prev,g,g.next)>=0)return!1;g=g.next}return!0}function W_(n,e,t,i){const s=n.prev,r=n,o=n.next;if(bt(s,r,o)>=0)return!1;const a=s.x,l=r.x,c=o.x,u=s.y,h=r.y,d=o.y,f=Math.min(a,l,c),g=Math.min(u,h,d),_=Math.max(a,l,c),m=Math.max(u,h,d),p=yc(f,g,e,t,i),C=yc(_,m,e,t,i);let M=n.prevZ,x=n.nextZ;for(;M&&M.z>=p&&x&&x.z<=C;){if(M.x>=f&&M.x<=_&&M.y>=g&&M.y<=m&&M!==s&&M!==o&&Ls(a,u,l,h,c,d,M.x,M.y)&&bt(M.prev,M,M.next)>=0||(M=M.prevZ,x.x>=f&&x.x<=_&&x.y>=g&&x.y<=m&&x!==s&&x!==o&&Ls(a,u,l,h,c,d,x.x,x.y)&&bt(x.prev,x,x.next)>=0))return!1;x=x.nextZ}for(;M&&M.z>=p;){if(M.x>=f&&M.x<=_&&M.y>=g&&M.y<=m&&M!==s&&M!==o&&Ls(a,u,l,h,c,d,M.x,M.y)&&bt(M.prev,M,M.next)>=0)return!1;M=M.prevZ}for(;x&&x.z<=C;){if(x.x>=f&&x.x<=_&&x.y>=g&&x.y<=m&&x!==s&&x!==o&&Ls(a,u,l,h,c,d,x.x,x.y)&&bt(x.prev,x,x.next)>=0)return!1;x=x.nextZ}return!0}function X_(n,e){let t=n;do{const i=t.prev,s=t.next.next;!ds(i,s)&&_d(i,t,t.next,s)&&qs(i,s)&&qs(s,i)&&(e.push(i.i,t.i,s.i),js(t),js(t.next),t=n=s),t=t.next}while(t!==n);return Ci(t)}function $_(n,e,t,i,s,r){let o=n;do{let a=o.next.next;for(;a!==o.prev;){if(o.i!==a.i&&tv(o,a)){let l=vd(o,a);o=Ci(o,o.next),l=Ci(l,l.next),Ys(o,e,t,i,s,r,0),Ys(l,e,t,i,s,r,0);return}a=a.next}o=o.next}while(o!==n)}function Y_(n,e,t,i){const s=[];for(let r=0,o=e.length;r<o;r++){const a=e[r]*i,l=r<o-1?e[r+1]*i:n.length,c=md(n,a,l,i,!1);c===c.next&&(c.steiner=!0),s.push(ev(c))}s.sort(q_);for(let r=0;r<s.length;r++)t=j_(s[r],t);return t}function q_(n,e){let t=n.x-e.x;if(t===0&&(t=n.y-e.y,t===0)){const i=(n.next.y-n.y)/(n.next.x-n.x),s=(e.next.y-e.y)/(e.next.x-e.x);t=i-s}return t}function j_(n,e){const t=Z_(n,e);if(!t)return e;const i=vd(t,n);return Ci(i,i.next),Ci(t,t.next)}function Z_(n,e){let t=e;const i=n.x,s=n.y;let r=-1/0,o;if(ds(n,t))return t;do{if(ds(n,t.next))return t.next;if(s<=t.y&&s>=t.next.y&&t.next.y!==t.y){const h=t.x+(s-t.y)*(t.next.x-t.x)/(t.next.y-t.y);if(h<=i&&h>r&&(r=h,o=t.x<t.next.x?t:t.next,h===i))return o}t=t.next}while(t!==e);if(!o)return null;const a=o,l=o.x,c=o.y;let u=1/0;t=o;do{if(i>=t.x&&t.x>=l&&i!==t.x&&gd(s<c?i:r,s,l,c,s<c?r:i,s,t.x,t.y)){const h=Math.abs(s-t.y)/(i-t.x);qs(t,n)&&(h<u||h===u&&(t.x>o.x||t.x===o.x&&J_(o,t)))&&(o=t,u=h)}t=t.next}while(t!==a);return o}function J_(n,e){return bt(n.prev,n,e.prev)<0&&bt(e.next,n,n.next)<0}function K_(n,e,t,i){let s=n;do s.z===0&&(s.z=yc(s.x,s.y,e,t,i)),s.prevZ=s.prev,s.nextZ=s.next,s=s.next;while(s!==n);s.prevZ.nextZ=null,s.prevZ=null,Q_(s)}function Q_(n){let e,t=1;do{let i=n,s;n=null;let r=null;for(e=0;i;){e++;let o=i,a=0;for(let c=0;c<t&&(a++,o=o.nextZ,!!o);c++);let l=t;for(;a>0||l>0&&o;)a!==0&&(l===0||!o||i.z<=o.z)?(s=i,i=i.nextZ,a--):(s=o,o=o.nextZ,l--),r?r.nextZ=s:n=s,s.prevZ=r,r=s;i=o}r.nextZ=null,t*=2}while(e>1);return n}function yc(n,e,t,i,s){return n=(n-t)*s|0,e=(e-i)*s|0,n=(n|n<<8)&16711935,n=(n|n<<4)&252645135,n=(n|n<<2)&858993459,n=(n|n<<1)&1431655765,e=(e|e<<8)&16711935,e=(e|e<<4)&252645135,e=(e|e<<2)&858993459,e=(e|e<<1)&1431655765,n|e<<1}function ev(n){let e=n,t=n;do(e.x<t.x||e.x===t.x&&e.y<t.y)&&(t=e),e=e.next;while(e!==n);return t}function gd(n,e,t,i,s,r,o,a){return(s-o)*(e-a)>=(n-o)*(r-a)&&(n-o)*(i-a)>=(t-o)*(e-a)&&(t-o)*(r-a)>=(s-o)*(i-a)}function Ls(n,e,t,i,s,r,o,a){return!(n===o&&e===a)&&gd(n,e,t,i,s,r,o,a)}function tv(n,e){return n.next.i!==e.i&&n.prev.i!==e.i&&!nv(n,e)&&(qs(n,e)&&qs(e,n)&&iv(n,e)&&(bt(n.prev,n,e.prev)||bt(n,e.prev,e))||ds(n,e)&&bt(n.prev,n,n.next)>0&&bt(e.prev,e,e.next)>0)}function bt(n,e,t){return(e.y-n.y)*(t.x-e.x)-(e.x-n.x)*(t.y-e.y)}function ds(n,e){return n.x===e.x&&n.y===e.y}function _d(n,e,t,i){const s=Fr(bt(n,e,t)),r=Fr(bt(n,e,i)),o=Fr(bt(t,i,n)),a=Fr(bt(t,i,e));return!!(s!==r&&o!==a||s===0&&Or(n,t,e)||r===0&&Or(n,i,e)||o===0&&Or(t,n,i)||a===0&&Or(t,e,i))}function Or(n,e,t){return e.x<=Math.max(n.x,t.x)&&e.x>=Math.min(n.x,t.x)&&e.y<=Math.max(n.y,t.y)&&e.y>=Math.min(n.y,t.y)}function Fr(n){return n>0?1:n<0?-1:0}function nv(n,e){let t=n;do{if(t.i!==n.i&&t.next.i!==n.i&&t.i!==e.i&&t.next.i!==e.i&&_d(t,t.next,n,e))return!0;t=t.next}while(t!==n);return!1}function qs(n,e){return bt(n.prev,n,n.next)<0?bt(n,e,n.next)>=0&&bt(n,n.prev,e)>=0:bt(n,e,n.prev)<0||bt(n,n.next,e)<0}function iv(n,e){let t=n,i=!1;const s=(n.x+e.x)/2,r=(n.y+e.y)/2;do t.y>r!=t.next.y>r&&t.next.y!==t.y&&s<(t.next.x-t.x)*(r-t.y)/(t.next.y-t.y)+t.x&&(i=!i),t=t.next;while(t!==n);return i}function vd(n,e){const t=Sc(n.i,n.x,n.y),i=Sc(e.i,e.x,e.y),s=n.next,r=e.prev;return n.next=e,e.prev=n,t.next=s,s.prev=t,i.next=t,t.prev=i,r.next=i,i.prev=r,i}function xu(n,e,t,i){const s=Sc(n,e,t);return i?(s.next=i.next,s.prev=i,i.next.prev=s,i.next=s):(s.prev=s,s.next=s),s}function js(n){n.next.prev=n.prev,n.prev.next=n.next,n.prevZ&&(n.prevZ.nextZ=n.nextZ),n.nextZ&&(n.nextZ.prevZ=n.prevZ)}function Sc(n,e,t){return{i:n,x:e,y:t,prev:null,next:null,z:0,prevZ:null,nextZ:null,steiner:!1}}function sv(n,e,t,i){let s=0;for(let r=e,o=t-i;r<t;r+=i)s+=(n[o]-n[r])*(n[r+1]+n[o+1]),o=r;return s}class rv{static triangulate(e,t,i=2){return G_(e,t,i)}}class Fs{static area(e){const t=e.length;let i=0;for(let s=t-1,r=0;r<t;s=r++)i+=e[s].x*e[r].y-e[r].x*e[s].y;return i*.5}static isClockWise(e){return Fs.area(e)<0}static triangulateShape(e,t){const i=[],s=[],r=[];yu(e),Su(i,e);let o=e.length;t.forEach(yu);for(let l=0;l<t.length;l++)s.push(o),o+=t[l].length,Su(i,t[l]);const a=rv.triangulate(i,s);for(let l=0;l<a.length;l+=3)r.push(a.slice(l,l+3));return r}}function yu(n){const e=n.length;e>2&&n[e-1].equals(n[0])&&n.pop()}function Su(n,e){for(let t=0;t<e.length;t++)n.push(e[t].x),n.push(e[t].y)}class ms extends it{constructor(e=1,t=1,i=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:i,heightSegments:s};const r=e/2,o=t/2,a=Math.floor(i),l=Math.floor(s),c=a+1,u=l+1,h=e/a,d=t/l,f=[],g=[],_=[],m=[];for(let p=0;p<u;p++){const C=p*d-o;for(let M=0;M<c;M++){const x=M*h-r;g.push(x,-C,0),_.push(0,0,1),m.push(M/a),m.push(1-p/l)}}for(let p=0;p<l;p++)for(let C=0;C<a;C++){const M=C+c*p,x=C+c*(p+1),P=C+1+c*(p+1),w=C+1+c*p;f.push(M,x,w),f.push(x,P,w)}this.setIndex(f),this.setAttribute("position",new gt(g,3)),this.setAttribute("normal",new gt(_,3)),this.setAttribute("uv",new gt(m,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new ms(e.width,e.height,e.widthSegments,e.heightSegments)}}class nr extends it{constructor(e=new Mo([new ue(0,.5),new ue(-.5,-.5),new ue(.5,-.5)]),t=12){super(),this.type="ShapeGeometry",this.parameters={shapes:e,curveSegments:t};const i=[],s=[],r=[],o=[];let a=0,l=0;if(Array.isArray(e)===!1)c(e);else for(let u=0;u<e.length;u++)c(e[u]),this.addGroup(a,l,u),a+=l,l=0;this.setIndex(i),this.setAttribute("position",new gt(s,3)),this.setAttribute("normal",new gt(r,3)),this.setAttribute("uv",new gt(o,2));function c(u){const h=s.length/3,d=u.extractPoints(t);let f=d.shape;const g=d.holes;Fs.isClockWise(f)===!1&&(f=f.reverse());for(let m=0,p=g.length;m<p;m++){const C=g[m];Fs.isClockWise(C)===!0&&(g[m]=C.reverse())}const _=Fs.triangulateShape(f,g);for(let m=0,p=g.length;m<p;m++){const C=g[m];f=f.concat(C)}for(let m=0,p=f.length;m<p;m++){const C=f[m];s.push(C.x,C.y,0),r.push(0,0,1),o.push(C.x,C.y)}for(let m=0,p=_.length;m<p;m++){const C=_[m],M=C[0]+h,x=C[1]+h,P=C[2]+h;i.push(M,x,P),l+=3}}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}toJSON(){const e=super.toJSON(),t=this.parameters.shapes;return ov(t,e)}static fromJSON(e,t){const i=[];for(let s=0,r=e.shapes.length;s<r;s++){const o=t[e.shapes[s]];i.push(o)}return new nr(i,e.curveSegments)}}function ov(n,e){if(e.shapes=[],Array.isArray(n))for(let t=0,i=n.length;t<i;t++){const s=n[t];e.shapes.push(s.uuid)}else e.shapes.push(n.uuid);return e}class fo extends it{constructor(e=1,t=32,i=16,s=0,r=Math.PI*2,o=0,a=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:e,widthSegments:t,heightSegments:i,phiStart:s,phiLength:r,thetaStart:o,thetaLength:a},t=Math.max(3,Math.floor(t)),i=Math.max(2,Math.floor(i));const l=Math.min(o+a,Math.PI);let c=0;const u=[],h=new R,d=new R,f=[],g=[],_=[],m=[];for(let p=0;p<=i;p++){const C=[],M=p/i;let x=0;p===0&&o===0?x=.5/t:p===i&&l===Math.PI&&(x=-.5/t);for(let P=0;P<=t;P++){const w=P/t;h.x=-e*Math.cos(s+w*r)*Math.sin(o+M*a),h.y=e*Math.cos(o+M*a),h.z=e*Math.sin(s+w*r)*Math.sin(o+M*a),g.push(h.x,h.y,h.z),d.copy(h).normalize(),_.push(d.x,d.y,d.z),m.push(w+x,1-M),C.push(c++)}u.push(C)}for(let p=0;p<i;p++)for(let C=0;C<t;C++){const M=u[p][C+1],x=u[p][C],P=u[p+1][C],w=u[p+1][C+1];(p!==0||o>0)&&f.push(M,x,w),(p!==i-1||l<Math.PI)&&f.push(x,P,w)}this.setIndex(f),this.setAttribute("position",new gt(g,3)),this.setAttribute("normal",new gt(_,3)),this.setAttribute("uv",new gt(m,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new fo(e.radius,e.widthSegments,e.heightSegments,e.phiStart,e.phiLength,e.thetaStart,e.thetaLength)}}class av extends ps{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Bg,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class cv extends ps{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}class xd extends Pn{constructor(e){super(),this.isLineDashedMaterial=!0,this.type="LineDashedMaterial",this.scale=1,this.dashSize=3,this.gapSize=1,this.setValues(e)}copy(e){return super.copy(e),this.scale=e.scale,this.dashSize=e.dashSize,this.gapSize=e.gapSize,this}}const ca={enabled:!1,files:{},add:function(n,e){this.enabled!==!1&&(this.files[n]=e)},get:function(n){if(this.enabled!==!1)return this.files[n]},remove:function(n){delete this.files[n]},clear:function(){this.files={}}};class lv{constructor(e,t,i){const s=this;let r=!1,o=0,a=0,l;const c=[];this.onStart=void 0,this.onLoad=e,this.onProgress=t,this.onError=i,this.abortController=new AbortController,this.itemStart=function(u){a++,r===!1&&s.onStart!==void 0&&s.onStart(u,o,a),r=!0},this.itemEnd=function(u){o++,s.onProgress!==void 0&&s.onProgress(u,o,a),o===a&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(u){s.onError!==void 0&&s.onError(u)},this.resolveURL=function(u){return l?l(u):u},this.setURLModifier=function(u){return l=u,this},this.addHandler=function(u,h){return c.push(u,h),this},this.removeHandler=function(u){const h=c.indexOf(u);return h!==-1&&c.splice(h,2),this},this.getHandler=function(u){for(let h=0,d=c.length;h<d;h+=2){const f=c[h],g=c[h+1];if(f.global&&(f.lastIndex=0),f.test(u))return g}return null},this.abort=function(){return this.abortController.abort(),this.abortController=new AbortController,this}}}const uv=new lv;class Yc{constructor(e){this.manager=e!==void 0?e:uv,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={}}load(){}loadAsync(e,t){const i=this;return new Promise(function(s,r){i.load(e,s,t,r)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}}Yc.DEFAULT_MATERIAL_NAME="__DEFAULT";const ji=new WeakMap;class hv extends Yc{constructor(e){super(e)}load(e,t,i,s){this.path!==void 0&&(e=this.path+e),e=this.manager.resolveURL(e);const r=this,o=ca.get(`image:${e}`);if(o!==void 0){if(o.complete===!0)r.manager.itemStart(e),setTimeout(function(){t&&t(o),r.manager.itemEnd(e)},0);else{let h=ji.get(o);h===void 0&&(h=[],ji.set(o,h)),h.push({onLoad:t,onError:s})}return o}const a=Ws("img");function l(){u(),t&&t(this);const h=ji.get(this)||[];for(let d=0;d<h.length;d++){const f=h[d];f.onLoad&&f.onLoad(this)}ji.delete(this),r.manager.itemEnd(e)}function c(h){u(),s&&s(h),ca.remove(`image:${e}`);const d=ji.get(this)||[];for(let f=0;f<d.length;f++){const g=d[f];g.onError&&g.onError(h)}ji.delete(this),r.manager.itemError(e),r.manager.itemEnd(e)}function u(){a.removeEventListener("load",l,!1),a.removeEventListener("error",c,!1)}return a.addEventListener("load",l,!1),a.addEventListener("error",c,!1),e.slice(0,5)!=="data:"&&this.crossOrigin!==void 0&&(a.crossOrigin=this.crossOrigin),ca.add(`image:${e}`,a),r.manager.itemStart(e),a.src=e,a}}class dv extends Yc{constructor(e){super(e)}load(e,t,i,s){const r=new Vt,o=new hv(this.manager);return o.setCrossOrigin(this.crossOrigin),o.setPath(this.path),o.load(e,function(a){r.image=a,r.needsUpdate=!0,t!==void 0&&t(r)},i,s),r}}class qc extends Lt{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new Qe(e),this.intensity=t}dispose(){}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){const t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,this.groundColor!==void 0&&(t.object.groundColor=this.groundColor.getHex()),this.distance!==void 0&&(t.object.distance=this.distance),this.angle!==void 0&&(t.object.angle=this.angle),this.decay!==void 0&&(t.object.decay=this.decay),this.penumbra!==void 0&&(t.object.penumbra=this.penumbra),this.shadow!==void 0&&(t.object.shadow=this.shadow.toJSON()),this.target!==void 0&&(t.object.target=this.target.uuid),t}}const la=new ft,bu=new R,Mu=new R;class yd{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new ue(512,512),this.mapType=Rn,this.map=null,this.mapPass=null,this.matrix=new ft,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new Wc,this._frameExtents=new ue(1,1),this._viewportCount=1,this._viewports=[new ct(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){const t=this.camera,i=this.matrix;bu.setFromMatrixPosition(e.matrixWorld),t.position.copy(bu),Mu.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(Mu),t.updateMatrixWorld(),la.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix(la,t.coordinateSystem,t.reversedDepth),t.reversedDepth?i.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):i.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),i.multiply(la)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this}clone(){return new this.constructor().copy(this)}toJSON(){const e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}}const wu=new ft,Ps=new R,ua=new R;class fv extends yd{constructor(){super(new sn(90,1,.5,500)),this.isPointLightShadow=!0,this._frameExtents=new ue(4,2),this._viewportCount=6,this._viewports=[new ct(2,1,1,1),new ct(0,1,1,1),new ct(3,1,1,1),new ct(1,1,1,1),new ct(3,0,1,1),new ct(1,0,1,1)],this._cubeDirections=[new R(1,0,0),new R(-1,0,0),new R(0,0,1),new R(0,0,-1),new R(0,1,0),new R(0,-1,0)],this._cubeUps=[new R(0,1,0),new R(0,1,0),new R(0,1,0),new R(0,1,0),new R(0,0,1),new R(0,0,-1)]}updateMatrices(e,t=0){const i=this.camera,s=this.matrix,r=e.distance||i.far;r!==i.far&&(i.far=r,i.updateProjectionMatrix()),Ps.setFromMatrixPosition(e.matrixWorld),i.position.copy(Ps),ua.copy(i.position),ua.add(this._cubeDirections[t]),i.up.copy(this._cubeUps[t]),i.lookAt(ua),i.updateMatrixWorld(),s.makeTranslation(-Ps.x,-Ps.y,-Ps.z),wu.multiplyMatrices(i.projectionMatrix,i.matrixWorldInverse),this._frustum.setFromProjectionMatrix(wu,i.coordinateSystem,i.reversedDepth)}}class pv extends qc{constructor(e,t,i=0,s=2){super(e,t),this.isPointLight=!0,this.type="PointLight",this.distance=i,this.decay=s,this.shadow=new fv}get power(){return this.intensity*4*Math.PI}set power(e){this.intensity=e/(4*Math.PI)}dispose(){this.shadow.dispose()}copy(e,t){return super.copy(e,t),this.distance=e.distance,this.decay=e.decay,this.shadow=e.shadow.clone(),this}}class jc extends sd{constructor(e=-1,t=1,i=1,s=-1,r=.1,o=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=i,this.bottom=s,this.near=r,this.far=o,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,i,s,r,o){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=i,this.view.offsetY=s,this.view.width=r,this.view.height=o,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),i=(this.right+this.left)/2,s=(this.top+this.bottom)/2;let r=i-e,o=i+e,a=s+t,l=s-t;if(this.view!==null&&this.view.enabled){const c=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=c*this.view.offsetX,o=r+c*this.view.width,a-=u*this.view.offsetY,l=a-u*this.view.height}this.projectionMatrix.makeOrthographic(r,o,a,l,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class mv extends yd{constructor(){super(new jc(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}}class gv extends qc{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(Lt.DEFAULT_UP),this.updateMatrix(),this.target=new Lt,this.shadow=new mv}dispose(){this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}}class _v extends qc{constructor(e,t){super(e,t),this.isAmbientLight=!0,this.type="AmbientLight"}}class vv extends sn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}class xv{constructor(e=!0){this.autoStart=e,this.startTime=0,this.oldTime=0,this.elapsedTime=0,this.running=!1}start(){this.startTime=performance.now(),this.oldTime=this.startTime,this.elapsedTime=0,this.running=!0}stop(){this.getElapsedTime(),this.running=!1,this.autoStart=!1}getElapsedTime(){return this.getDelta(),this.elapsedTime}getDelta(){let e=0;if(this.autoStart&&!this.running)return this.start(),0;if(this.running){const t=performance.now();e=(t-this.oldTime)/1e3,this.oldTime=t,this.elapsedTime+=e}return e}}const Eu=new ft;class yv{constructor(e,t,i=0,s=1/0){this.ray=new yo(e,t),this.near=i,this.far=s,this.camera=null,this.layers=new Hc,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(e,t){this.ray.set(e,t)}setFromCamera(e,t){t.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t):t.isOrthographicCamera?(this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t):console.error("THREE.Raycaster: Unsupported camera type: "+t.type)}setFromXRController(e){return Eu.identity().extractRotation(e.matrixWorld),this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(Eu),this}intersectObject(e,t=!0,i=[]){return bc(e,this,i,t),i.sort(Tu),i}intersectObjects(e,t=!0,i=[]){for(let s=0,r=e.length;s<r;s++)bc(e[s],this,i,t);return i.sort(Tu),i}}function Tu(n,e){return n.distance-e.distance}function bc(n,e,t,i){let s=!0;if(n.layers.test(e.layers)&&n.raycast(e,t)===!1&&(s=!1),s===!0&&i===!0){const r=n.children;for(let o=0,a=r.length;o<a;o++)bc(r[o],e,t,!0)}}class Au{constructor(e=1,t=0,i=0){this.radius=e,this.phi=t,this.theta=i}set(e,t,i){return this.radius=e,this.phi=t,this.theta=i,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=Ze(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,i){return this.radius=Math.sqrt(e*e+t*t+i*i),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,i),this.phi=Math.acos(Ze(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}const Cu=new R;let kr,ha;class Ru extends Lt{constructor(e=new R(0,0,1),t=new R(0,0,0),i=1,s=16776960,r=i*.2,o=r*.2){super(),this.type="ArrowHelper",kr===void 0&&(kr=new it,kr.setAttribute("position",new gt([0,0,0,0,1,0],3)),ha=new bo(.5,1,5,1),ha.translate(0,-.5,0)),this.position.copy(t),this.line=new Ln(kr,new Pn({color:s,toneMapped:!1})),this.line.matrixAutoUpdate=!1,this.add(this.line),this.cone=new xt(ha,new qt({color:s,toneMapped:!1})),this.cone.matrixAutoUpdate=!1,this.add(this.cone),this.setDirection(e),this.setLength(i,r,o)}setDirection(e){if(e.y>.99999)this.quaternion.set(0,0,0,1);else if(e.y<-.99999)this.quaternion.set(1,0,0,0);else{Cu.set(e.z,0,-e.x).normalize();const t=Math.acos(e.y);this.quaternion.setFromAxisAngle(Cu,t)}}setLength(e,t=e*.2,i=t*.2){this.line.scale.set(1,Math.max(1e-4,e-t),1),this.line.updateMatrix(),this.cone.scale.set(i,t,i),this.cone.position.y=e,this.cone.updateMatrix()}setColor(e){this.line.material.color.set(e),this.cone.material.color.set(e)}copy(e){return super.copy(e,!1),this.line.copy(e.line),this.cone.copy(e.cone),this}dispose(){this.line.geometry.dispose(),this.line.material.dispose(),this.cone.geometry.dispose(),this.cone.material.dispose()}}class Sv extends Ri{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){console.warn("THREE.Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}}function Pu(n,e,t,i){const s=bv(i);switch(t){case Yh:return n*e;case jh:return n*e/s.components*s.byteLength;case Fc:return n*e/s.components*s.byteLength;case Zh:return n*e*2/s.components*s.byteLength;case kc:return n*e*2/s.components*s.byteLength;case qh:return n*e*3/s.components*s.byteLength;case vn:return n*e*4/s.components*s.byteLength;case zc:return n*e*4/s.components*s.byteLength;case qr:case jr:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*8;case Zr:case Jr:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case Xa:case Ya:return Math.max(n,16)*Math.max(e,8)/4;case Wa:case $a:return Math.max(n,8)*Math.max(e,8)/2;case qa:case ja:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*8;case Za:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case Ja:return Math.floor((n+3)/4)*Math.floor((e+3)/4)*16;case Ka:return Math.floor((n+4)/5)*Math.floor((e+3)/4)*16;case Qa:return Math.floor((n+4)/5)*Math.floor((e+4)/5)*16;case ec:return Math.floor((n+5)/6)*Math.floor((e+4)/5)*16;case tc:return Math.floor((n+5)/6)*Math.floor((e+5)/6)*16;case nc:return Math.floor((n+7)/8)*Math.floor((e+4)/5)*16;case ic:return Math.floor((n+7)/8)*Math.floor((e+5)/6)*16;case sc:return Math.floor((n+7)/8)*Math.floor((e+7)/8)*16;case rc:return Math.floor((n+9)/10)*Math.floor((e+4)/5)*16;case oc:return Math.floor((n+9)/10)*Math.floor((e+5)/6)*16;case ac:return Math.floor((n+9)/10)*Math.floor((e+7)/8)*16;case cc:return Math.floor((n+9)/10)*Math.floor((e+9)/10)*16;case lc:return Math.floor((n+11)/12)*Math.floor((e+9)/10)*16;case uc:return Math.floor((n+11)/12)*Math.floor((e+11)/12)*16;case hc:case dc:case fc:return Math.ceil(n/4)*Math.ceil(e/4)*16;case pc:case mc:return Math.ceil(n/4)*Math.ceil(e/4)*8;case gc:case _c:return Math.ceil(n/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function bv(n){switch(n){case Rn:case Vh:return{byteLength:1,components:1};case Bs:case Wh:case Ks:return{byteLength:2,components:1};case Uc:case Oc:return{byteLength:2,components:4};case Ei:case Nc:case Wn:return{byteLength:4,components:1};case Xh:case $h:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${n}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:Ic}}));typeof window<"u"&&(window.__THREE__?console.warn("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=Ic);function Sd(){let n=null,e=!1,t=null,i=null;function s(r,o){t(r,o),i=n.requestAnimationFrame(s)}return{start:function(){e!==!0&&t!==null&&(i=n.requestAnimationFrame(s),e=!0)},stop:function(){n.cancelAnimationFrame(i),e=!1},setAnimationLoop:function(r){t=r},setContext:function(r){n=r}}}function Mv(n){const e=new WeakMap;function t(a,l){const c=a.array,u=a.usage,h=c.byteLength,d=n.createBuffer();n.bindBuffer(l,d),n.bufferData(l,c,u),a.onUploadCallback();let f;if(c instanceof Float32Array)f=n.FLOAT;else if(typeof Float16Array<"u"&&c instanceof Float16Array)f=n.HALF_FLOAT;else if(c instanceof Uint16Array)a.isFloat16BufferAttribute?f=n.HALF_FLOAT:f=n.UNSIGNED_SHORT;else if(c instanceof Int16Array)f=n.SHORT;else if(c instanceof Uint32Array)f=n.UNSIGNED_INT;else if(c instanceof Int32Array)f=n.INT;else if(c instanceof Int8Array)f=n.BYTE;else if(c instanceof Uint8Array)f=n.UNSIGNED_BYTE;else if(c instanceof Uint8ClampedArray)f=n.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+c);return{buffer:d,type:f,bytesPerElement:c.BYTES_PER_ELEMENT,version:a.version,size:h}}function i(a,l,c){const u=l.array,h=l.updateRanges;if(n.bindBuffer(c,a),h.length===0)n.bufferSubData(c,0,u);else{h.sort((f,g)=>f.start-g.start);let d=0;for(let f=1;f<h.length;f++){const g=h[d],_=h[f];_.start<=g.start+g.count+1?g.count=Math.max(g.count,_.start+_.count-g.start):(++d,h[d]=_)}h.length=d+1;for(let f=0,g=h.length;f<g;f++){const _=h[f];n.bufferSubData(c,_.start*u.BYTES_PER_ELEMENT,u,_.start,_.count)}l.clearUpdateRanges()}l.onUploadCallback()}function s(a){return a.isInterleavedBufferAttribute&&(a=a.data),e.get(a)}function r(a){a.isInterleavedBufferAttribute&&(a=a.data);const l=e.get(a);l&&(n.deleteBuffer(l.buffer),e.delete(a))}function o(a,l){if(a.isInterleavedBufferAttribute&&(a=a.data),a.isGLBufferAttribute){const u=e.get(a);(!u||u.version<a.version)&&e.set(a,{buffer:a.buffer,type:a.type,bytesPerElement:a.elementSize,version:a.version});return}const c=e.get(a);if(c===void 0)e.set(a,t(a,l));else if(c.version<a.version){if(c.size!==a.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");i(c.buffer,a,l),c.version=a.version}}return{get:s,remove:r,update:o}}var wv=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Ev=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Tv=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Av=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Cv=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Rv=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Pv=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Lv=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Dv=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`,Iv=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,Nv=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Uv=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Ov=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Fv=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,kv=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,zv=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,Bv=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Hv=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Gv=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Vv=`#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`,Wv=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`,Xv=`#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`,$v=`#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`,Yv=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,qv=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,jv=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,Zv=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,Jv=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Kv=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Qv=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,e0="gl_FragColor = linearToOutputTexel( gl_FragColor );",t0=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,n0=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`,i0=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`,s0=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,r0=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,o0=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,a0=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,c0=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,l0=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,u0=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,h0=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,d0=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,f0=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,p0=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,m0=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,g0=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,_0=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,v0=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,x0=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,y0=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,S0=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,b0=`struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,M0=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,w0=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,E0=`#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,T0=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,A0=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,C0=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,R0=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,P0=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,L0=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,D0=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,I0=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,N0=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,U0=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,O0=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,F0=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,k0=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,z0=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,B0=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,H0=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,G0=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,V0=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,W0=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,X0=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,$0=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Y0=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,q0=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,j0=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,Z0=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,J0=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,K0=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`,Q0=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,ex=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,tx=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,nx=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,ix=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,sx=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,rx=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		float depth = unpackRGBAToDepth( texture2D( depths, uv ) );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			return step( depth, compare );
		#else
			return step( compare, depth );
		#endif
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow( sampler2D shadow, vec2 uv, float compare ) {
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		#ifdef USE_REVERSED_DEPTH_BUFFER
			float hard_shadow = step( distribution.x, compare );
		#else
			float hard_shadow = step( compare, distribution.x );
		#endif
		if ( hard_shadow != 1.0 ) {
			float distance = compare - distribution.x;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`,ox=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,ax=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,cx=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,lx=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,ux=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,hx=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,dx=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,fx=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,px=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,mx=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,gx=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,_x=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,vx=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,xx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,yx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Sx=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,bx=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const Mx=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,wx=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ex=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Tx=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ax=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Cx=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Rx=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Px=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Lx=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Dx=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`,Ix=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Nx=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ux=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Ox=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Fx=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,kx=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,zx=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Bx=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Hx=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Gx=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Vx=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Wx=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Xx=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,$x=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Yx=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,qx=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,jx=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Zx=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Jx=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Kx=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Qx=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,ey=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ty=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,ny=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,$e={alphahash_fragment:wv,alphahash_pars_fragment:Ev,alphamap_fragment:Tv,alphamap_pars_fragment:Av,alphatest_fragment:Cv,alphatest_pars_fragment:Rv,aomap_fragment:Pv,aomap_pars_fragment:Lv,batching_pars_vertex:Dv,batching_vertex:Iv,begin_vertex:Nv,beginnormal_vertex:Uv,bsdfs:Ov,iridescence_fragment:Fv,bumpmap_pars_fragment:kv,clipping_planes_fragment:zv,clipping_planes_pars_fragment:Bv,clipping_planes_pars_vertex:Hv,clipping_planes_vertex:Gv,color_fragment:Vv,color_pars_fragment:Wv,color_pars_vertex:Xv,color_vertex:$v,common:Yv,cube_uv_reflection_fragment:qv,defaultnormal_vertex:jv,displacementmap_pars_vertex:Zv,displacementmap_vertex:Jv,emissivemap_fragment:Kv,emissivemap_pars_fragment:Qv,colorspace_fragment:e0,colorspace_pars_fragment:t0,envmap_fragment:n0,envmap_common_pars_fragment:i0,envmap_pars_fragment:s0,envmap_pars_vertex:r0,envmap_physical_pars_fragment:g0,envmap_vertex:o0,fog_vertex:a0,fog_pars_vertex:c0,fog_fragment:l0,fog_pars_fragment:u0,gradientmap_pars_fragment:h0,lightmap_pars_fragment:d0,lights_lambert_fragment:f0,lights_lambert_pars_fragment:p0,lights_pars_begin:m0,lights_toon_fragment:_0,lights_toon_pars_fragment:v0,lights_phong_fragment:x0,lights_phong_pars_fragment:y0,lights_physical_fragment:S0,lights_physical_pars_fragment:b0,lights_fragment_begin:M0,lights_fragment_maps:w0,lights_fragment_end:E0,logdepthbuf_fragment:T0,logdepthbuf_pars_fragment:A0,logdepthbuf_pars_vertex:C0,logdepthbuf_vertex:R0,map_fragment:P0,map_pars_fragment:L0,map_particle_fragment:D0,map_particle_pars_fragment:I0,metalnessmap_fragment:N0,metalnessmap_pars_fragment:U0,morphinstance_vertex:O0,morphcolor_vertex:F0,morphnormal_vertex:k0,morphtarget_pars_vertex:z0,morphtarget_vertex:B0,normal_fragment_begin:H0,normal_fragment_maps:G0,normal_pars_fragment:V0,normal_pars_vertex:W0,normal_vertex:X0,normalmap_pars_fragment:$0,clearcoat_normal_fragment_begin:Y0,clearcoat_normal_fragment_maps:q0,clearcoat_pars_fragment:j0,iridescence_pars_fragment:Z0,opaque_fragment:J0,packing:K0,premultiplied_alpha_fragment:Q0,project_vertex:ex,dithering_fragment:tx,dithering_pars_fragment:nx,roughnessmap_fragment:ix,roughnessmap_pars_fragment:sx,shadowmap_pars_fragment:rx,shadowmap_pars_vertex:ox,shadowmap_vertex:ax,shadowmask_pars_fragment:cx,skinbase_vertex:lx,skinning_pars_vertex:ux,skinning_vertex:hx,skinnormal_vertex:dx,specularmap_fragment:fx,specularmap_pars_fragment:px,tonemapping_fragment:mx,tonemapping_pars_fragment:gx,transmission_fragment:_x,transmission_pars_fragment:vx,uv_pars_fragment:xx,uv_pars_vertex:yx,uv_vertex:Sx,worldpos_vertex:bx,background_vert:Mx,background_frag:wx,backgroundCube_vert:Ex,backgroundCube_frag:Tx,cube_vert:Ax,cube_frag:Cx,depth_vert:Rx,depth_frag:Px,distanceRGBA_vert:Lx,distanceRGBA_frag:Dx,equirect_vert:Ix,equirect_frag:Nx,linedashed_vert:Ux,linedashed_frag:Ox,meshbasic_vert:Fx,meshbasic_frag:kx,meshlambert_vert:zx,meshlambert_frag:Bx,meshmatcap_vert:Hx,meshmatcap_frag:Gx,meshnormal_vert:Vx,meshnormal_frag:Wx,meshphong_vert:Xx,meshphong_frag:$x,meshphysical_vert:Yx,meshphysical_frag:qx,meshtoon_vert:jx,meshtoon_frag:Zx,points_vert:Jx,points_frag:Kx,shadow_vert:Qx,shadow_frag:ey,sprite_vert:ty,sprite_frag:ny},de={common:{diffuse:{value:new Qe(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Ve},alphaMap:{value:null},alphaMapTransform:{value:new Ve},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Ve}},envmap:{envMap:{value:null},envMapRotation:{value:new Ve},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Ve}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Ve}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Ve},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Ve},normalScale:{value:new ue(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Ve},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Ve}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Ve}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Ve}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Qe(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMap:{value:[]},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotShadowMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMap:{value:[]},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new Qe(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Ve},alphaTest:{value:0},uvTransform:{value:new Ve}},sprite:{diffuse:{value:new Qe(16777215)},opacity:{value:1},center:{value:new ue(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Ve},alphaMap:{value:null},alphaMapTransform:{value:new Ve},alphaTest:{value:0}}},Mn={basic:{uniforms:$t([de.common,de.specularmap,de.envmap,de.aomap,de.lightmap,de.fog]),vertexShader:$e.meshbasic_vert,fragmentShader:$e.meshbasic_frag},lambert:{uniforms:$t([de.common,de.specularmap,de.envmap,de.aomap,de.lightmap,de.emissivemap,de.bumpmap,de.normalmap,de.displacementmap,de.fog,de.lights,{emissive:{value:new Qe(0)}}]),vertexShader:$e.meshlambert_vert,fragmentShader:$e.meshlambert_frag},phong:{uniforms:$t([de.common,de.specularmap,de.envmap,de.aomap,de.lightmap,de.emissivemap,de.bumpmap,de.normalmap,de.displacementmap,de.fog,de.lights,{emissive:{value:new Qe(0)},specular:{value:new Qe(1118481)},shininess:{value:30}}]),vertexShader:$e.meshphong_vert,fragmentShader:$e.meshphong_frag},standard:{uniforms:$t([de.common,de.envmap,de.aomap,de.lightmap,de.emissivemap,de.bumpmap,de.normalmap,de.displacementmap,de.roughnessmap,de.metalnessmap,de.fog,de.lights,{emissive:{value:new Qe(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:$e.meshphysical_vert,fragmentShader:$e.meshphysical_frag},toon:{uniforms:$t([de.common,de.aomap,de.lightmap,de.emissivemap,de.bumpmap,de.normalmap,de.displacementmap,de.gradientmap,de.fog,de.lights,{emissive:{value:new Qe(0)}}]),vertexShader:$e.meshtoon_vert,fragmentShader:$e.meshtoon_frag},matcap:{uniforms:$t([de.common,de.bumpmap,de.normalmap,de.displacementmap,de.fog,{matcap:{value:null}}]),vertexShader:$e.meshmatcap_vert,fragmentShader:$e.meshmatcap_frag},points:{uniforms:$t([de.points,de.fog]),vertexShader:$e.points_vert,fragmentShader:$e.points_frag},dashed:{uniforms:$t([de.common,de.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:$e.linedashed_vert,fragmentShader:$e.linedashed_frag},depth:{uniforms:$t([de.common,de.displacementmap]),vertexShader:$e.depth_vert,fragmentShader:$e.depth_frag},normal:{uniforms:$t([de.common,de.bumpmap,de.normalmap,de.displacementmap,{opacity:{value:1}}]),vertexShader:$e.meshnormal_vert,fragmentShader:$e.meshnormal_frag},sprite:{uniforms:$t([de.sprite,de.fog]),vertexShader:$e.sprite_vert,fragmentShader:$e.sprite_frag},background:{uniforms:{uvTransform:{value:new Ve},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:$e.background_vert,fragmentShader:$e.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Ve}},vertexShader:$e.backgroundCube_vert,fragmentShader:$e.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:$e.cube_vert,fragmentShader:$e.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:$e.equirect_vert,fragmentShader:$e.equirect_frag},distanceRGBA:{uniforms:$t([de.common,de.displacementmap,{referencePosition:{value:new R},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:$e.distanceRGBA_vert,fragmentShader:$e.distanceRGBA_frag},shadow:{uniforms:$t([de.lights,de.fog,{color:{value:new Qe(0)},opacity:{value:1}}]),vertexShader:$e.shadow_vert,fragmentShader:$e.shadow_frag}};Mn.physical={uniforms:$t([Mn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Ve},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Ve},clearcoatNormalScale:{value:new ue(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Ve},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Ve},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Ve},sheen:{value:0},sheenColor:{value:new Qe(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Ve},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Ve},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Ve},transmissionSamplerSize:{value:new ue},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Ve},attenuationDistance:{value:0},attenuationColor:{value:new Qe(0)},specularColor:{value:new Qe(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Ve},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Ve},anisotropyVector:{value:new ue},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Ve}}]),vertexShader:$e.meshphysical_vert,fragmentShader:$e.meshphysical_frag};const zr={r:0,b:0,g:0},mi=new Yn,iy=new ft;function sy(n,e,t,i,s,r,o){const a=new Qe(0);let l=r===!0?0:1,c,u,h=null,d=0,f=null;function g(M){let x=M.isScene===!0?M.background:null;return x&&x.isTexture&&(x=(M.backgroundBlurriness>0?t:e).get(x)),x}function _(M){let x=!1;const P=g(M);P===null?p(a,l):P&&P.isColor&&(p(P,1),x=!0);const w=n.xr.getEnvironmentBlendMode();w==="additive"?i.buffers.color.setClear(0,0,0,1,o):w==="alpha-blend"&&i.buffers.color.setClear(0,0,0,0,o),(n.autoClear||x)&&(i.buffers.depth.setTest(!0),i.buffers.depth.setMask(!0),i.buffers.color.setMask(!0),n.clear(n.autoClearColor,n.autoClearDepth,n.autoClearStencil))}function m(M,x){const P=g(x);P&&(P.isCubeTexture||P.mapping===vo)?(u===void 0&&(u=new xt(new er(1,1,1),new ai({name:"BackgroundCubeMaterial",uniforms:hs(Mn.backgroundCube.uniforms),vertexShader:Mn.backgroundCube.vertexShader,fragmentShader:Mn.backgroundCube.fragmentShader,side:Jt,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),u.geometry.deleteAttribute("normal"),u.geometry.deleteAttribute("uv"),u.onBeforeRender=function(w,T,L){this.matrixWorld.copyPosition(L.matrixWorld)},Object.defineProperty(u.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),s.update(u)),mi.copy(x.backgroundRotation),mi.x*=-1,mi.y*=-1,mi.z*=-1,P.isCubeTexture&&P.isRenderTargetTexture===!1&&(mi.y*=-1,mi.z*=-1),u.material.uniforms.envMap.value=P,u.material.uniforms.flipEnvMap.value=P.isCubeTexture&&P.isRenderTargetTexture===!1?-1:1,u.material.uniforms.backgroundBlurriness.value=x.backgroundBlurriness,u.material.uniforms.backgroundIntensity.value=x.backgroundIntensity,u.material.uniforms.backgroundRotation.value.setFromMatrix4(iy.makeRotationFromEuler(mi)),u.material.toneMapped=tt.getTransfer(P.colorSpace)!==rt,(h!==P||d!==P.version||f!==n.toneMapping)&&(u.material.needsUpdate=!0,h=P,d=P.version,f=n.toneMapping),u.layers.enableAll(),M.unshift(u,u.geometry,u.material,0,0,null)):P&&P.isTexture&&(c===void 0&&(c=new xt(new ms(2,2),new ai({name:"BackgroundMaterial",uniforms:hs(Mn.background.uniforms),vertexShader:Mn.background.vertexShader,fragmentShader:Mn.background.fragmentShader,side:oi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),s.update(c)),c.material.uniforms.t2D.value=P,c.material.uniforms.backgroundIntensity.value=x.backgroundIntensity,c.material.toneMapped=tt.getTransfer(P.colorSpace)!==rt,P.matrixAutoUpdate===!0&&P.updateMatrix(),c.material.uniforms.uvTransform.value.copy(P.matrix),(h!==P||d!==P.version||f!==n.toneMapping)&&(c.material.needsUpdate=!0,h=P,d=P.version,f=n.toneMapping),c.layers.enableAll(),M.unshift(c,c.geometry,c.material,0,0,null))}function p(M,x){M.getRGB(zr,id(n)),i.buffers.color.setClear(zr.r,zr.g,zr.b,x,o)}function C(){u!==void 0&&(u.geometry.dispose(),u.material.dispose(),u=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(M,x=1){a.set(M),l=x,p(a,l)},getClearAlpha:function(){return l},setClearAlpha:function(M){l=M,p(a,l)},render:_,addToRenderList:m,dispose:C}}function ry(n,e){const t=n.getParameter(n.MAX_VERTEX_ATTRIBS),i={},s=d(null);let r=s,o=!1;function a(b,N,O,G,V){let X=!1;const q=h(G,O,N);r!==q&&(r=q,c(r.object)),X=f(b,G,O,V),X&&g(b,G,O,V),V!==null&&e.update(V,n.ELEMENT_ARRAY_BUFFER),(X||o)&&(o=!1,x(b,N,O,G),V!==null&&n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,e.get(V).buffer))}function l(){return n.createVertexArray()}function c(b){return n.bindVertexArray(b)}function u(b){return n.deleteVertexArray(b)}function h(b,N,O){const G=O.wireframe===!0;let V=i[b.id];V===void 0&&(V={},i[b.id]=V);let X=V[N.id];X===void 0&&(X={},V[N.id]=X);let q=X[G];return q===void 0&&(q=d(l()),X[G]=q),q}function d(b){const N=[],O=[],G=[];for(let V=0;V<t;V++)N[V]=0,O[V]=0,G[V]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:N,enabledAttributes:O,attributeDivisors:G,object:b,attributes:{},index:null}}function f(b,N,O,G){const V=r.attributes,X=N.attributes;let q=0;const J=O.getAttributes();for(const z in J)if(J[z].location>=0){const D=V[z];let W=X[z];if(W===void 0&&(z==="instanceMatrix"&&b.instanceMatrix&&(W=b.instanceMatrix),z==="instanceColor"&&b.instanceColor&&(W=b.instanceColor)),D===void 0||D.attribute!==W||W&&D.data!==W.data)return!0;q++}return r.attributesNum!==q||r.index!==G}function g(b,N,O,G){const V={},X=N.attributes;let q=0;const J=O.getAttributes();for(const z in J)if(J[z].location>=0){let D=X[z];D===void 0&&(z==="instanceMatrix"&&b.instanceMatrix&&(D=b.instanceMatrix),z==="instanceColor"&&b.instanceColor&&(D=b.instanceColor));const W={};W.attribute=D,D&&D.data&&(W.data=D.data),V[z]=W,q++}r.attributes=V,r.attributesNum=q,r.index=G}function _(){const b=r.newAttributes;for(let N=0,O=b.length;N<O;N++)b[N]=0}function m(b){p(b,0)}function p(b,N){const O=r.newAttributes,G=r.enabledAttributes,V=r.attributeDivisors;O[b]=1,G[b]===0&&(n.enableVertexAttribArray(b),G[b]=1),V[b]!==N&&(n.vertexAttribDivisor(b,N),V[b]=N)}function C(){const b=r.newAttributes,N=r.enabledAttributes;for(let O=0,G=N.length;O<G;O++)N[O]!==b[O]&&(n.disableVertexAttribArray(O),N[O]=0)}function M(b,N,O,G,V,X,q){q===!0?n.vertexAttribIPointer(b,N,O,V,X):n.vertexAttribPointer(b,N,O,G,V,X)}function x(b,N,O,G){_();const V=G.attributes,X=O.getAttributes(),q=N.defaultAttributeValues;for(const J in X){const z=X[J];if(z.location>=0){let H=V[J];if(H===void 0&&(J==="instanceMatrix"&&b.instanceMatrix&&(H=b.instanceMatrix),J==="instanceColor"&&b.instanceColor&&(H=b.instanceColor)),H!==void 0){const D=H.normalized,W=H.itemSize,Q=e.get(H);if(Q===void 0)continue;const me=Q.buffer,De=Q.type,Ie=Q.bytesPerElement,j=De===n.INT||De===n.UNSIGNED_INT||H.gpuType===Nc;if(H.isInterleavedBufferAttribute){const te=H.data,xe=te.stride,Fe=H.offset;if(te.isInstancedInterleavedBuffer){for(let Te=0;Te<z.locationSize;Te++)p(z.location+Te,te.meshPerAttribute);b.isInstancedMesh!==!0&&G._maxInstanceCount===void 0&&(G._maxInstanceCount=te.meshPerAttribute*te.count)}else for(let Te=0;Te<z.locationSize;Te++)m(z.location+Te);n.bindBuffer(n.ARRAY_BUFFER,me);for(let Te=0;Te<z.locationSize;Te++)M(z.location+Te,W/z.locationSize,De,D,xe*Ie,(Fe+W/z.locationSize*Te)*Ie,j)}else{if(H.isInstancedBufferAttribute){for(let te=0;te<z.locationSize;te++)p(z.location+te,H.meshPerAttribute);b.isInstancedMesh!==!0&&G._maxInstanceCount===void 0&&(G._maxInstanceCount=H.meshPerAttribute*H.count)}else for(let te=0;te<z.locationSize;te++)m(z.location+te);n.bindBuffer(n.ARRAY_BUFFER,me);for(let te=0;te<z.locationSize;te++)M(z.location+te,W/z.locationSize,De,D,W*Ie,W/z.locationSize*te*Ie,j)}}else if(q!==void 0){const D=q[J];if(D!==void 0)switch(D.length){case 2:n.vertexAttrib2fv(z.location,D);break;case 3:n.vertexAttrib3fv(z.location,D);break;case 4:n.vertexAttrib4fv(z.location,D);break;default:n.vertexAttrib1fv(z.location,D)}}}}C()}function P(){L();for(const b in i){const N=i[b];for(const O in N){const G=N[O];for(const V in G)u(G[V].object),delete G[V];delete N[O]}delete i[b]}}function w(b){if(i[b.id]===void 0)return;const N=i[b.id];for(const O in N){const G=N[O];for(const V in G)u(G[V].object),delete G[V];delete N[O]}delete i[b.id]}function T(b){for(const N in i){const O=i[N];if(O[b.id]===void 0)continue;const G=O[b.id];for(const V in G)u(G[V].object),delete G[V];delete O[b.id]}}function L(){y(),o=!0,r!==s&&(r=s,c(r.object))}function y(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:a,reset:L,resetDefaultState:y,dispose:P,releaseStatesOfGeometry:w,releaseStatesOfProgram:T,initAttributes:_,enableAttribute:m,disableUnusedAttributes:C}}function oy(n,e,t){let i;function s(c){i=c}function r(c,u){n.drawArrays(i,c,u),t.update(u,i,1)}function o(c,u,h){h!==0&&(n.drawArraysInstanced(i,c,u,h),t.update(u,i,h))}function a(c,u,h){if(h===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(i,c,0,u,0,h);let f=0;for(let g=0;g<h;g++)f+=u[g];t.update(f,i,1)}function l(c,u,h,d){if(h===0)return;const f=e.get("WEBGL_multi_draw");if(f===null)for(let g=0;g<c.length;g++)o(c[g],u[g],d[g]);else{f.multiDrawArraysInstancedWEBGL(i,c,0,u,0,d,0,h);let g=0;for(let _=0;_<h;_++)g+=u[_]*d[_];t.update(g,i,1)}}this.setMode=s,this.render=r,this.renderInstances=o,this.renderMultiDraw=a,this.renderMultiDrawInstances=l}function ay(n,e,t,i){let s;function r(){if(s!==void 0)return s;if(e.has("EXT_texture_filter_anisotropic")===!0){const T=e.get("EXT_texture_filter_anisotropic");s=n.getParameter(T.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function o(T){return!(T!==vn&&i.convert(T)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_FORMAT))}function a(T){const L=T===Ks&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(T!==Rn&&i.convert(T)!==n.getParameter(n.IMPLEMENTATION_COLOR_READ_TYPE)&&T!==Wn&&!L)}function l(T){if(T==="highp"){if(n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.HIGH_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.HIGH_FLOAT).precision>0)return"highp";T="mediump"}return T==="mediump"&&n.getShaderPrecisionFormat(n.VERTEX_SHADER,n.MEDIUM_FLOAT).precision>0&&n.getShaderPrecisionFormat(n.FRAGMENT_SHADER,n.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let c=t.precision!==void 0?t.precision:"highp";const u=l(c);u!==c&&(console.warn("THREE.WebGLRenderer:",c,"not supported, using",u,"instead."),c=u);const h=t.logarithmicDepthBuffer===!0,d=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),f=n.getParameter(n.MAX_TEXTURE_IMAGE_UNITS),g=n.getParameter(n.MAX_VERTEX_TEXTURE_IMAGE_UNITS),_=n.getParameter(n.MAX_TEXTURE_SIZE),m=n.getParameter(n.MAX_CUBE_MAP_TEXTURE_SIZE),p=n.getParameter(n.MAX_VERTEX_ATTRIBS),C=n.getParameter(n.MAX_VERTEX_UNIFORM_VECTORS),M=n.getParameter(n.MAX_VARYING_VECTORS),x=n.getParameter(n.MAX_FRAGMENT_UNIFORM_VECTORS),P=g>0,w=n.getParameter(n.MAX_SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:l,textureFormatReadable:o,textureTypeReadable:a,precision:c,logarithmicDepthBuffer:h,reversedDepthBuffer:d,maxTextures:f,maxVertexTextures:g,maxTextureSize:_,maxCubemapSize:m,maxAttributes:p,maxVertexUniforms:C,maxVaryings:M,maxFragmentUniforms:x,vertexTextures:P,maxSamples:w}}function cy(n){const e=this;let t=null,i=0,s=!1,r=!1;const o=new Gn,a=new Ve,l={value:null,needsUpdate:!1};this.uniform=l,this.numPlanes=0,this.numIntersection=0,this.init=function(h,d){const f=h.length!==0||d||i!==0||s;return s=d,i=h.length,f},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(h,d){t=u(h,d,0)},this.setState=function(h,d,f){const g=h.clippingPlanes,_=h.clipIntersection,m=h.clipShadows,p=n.get(h);if(!s||g===null||g.length===0||r&&!m)r?u(null):c();else{const C=r?0:i,M=C*4;let x=p.clippingState||null;l.value=x,x=u(g,d,M,f);for(let P=0;P!==M;++P)x[P]=t[P];p.clippingState=x,this.numIntersection=_?this.numPlanes:0,this.numPlanes+=C}};function c(){l.value!==t&&(l.value=t,l.needsUpdate=i>0),e.numPlanes=i,e.numIntersection=0}function u(h,d,f,g){const _=h!==null?h.length:0;let m=null;if(_!==0){if(m=l.value,g!==!0||m===null){const p=f+_*4,C=d.matrixWorldInverse;a.getNormalMatrix(C),(m===null||m.length<p)&&(m=new Float32Array(p));for(let M=0,x=f;M!==_;++M,x+=4)o.copy(h[M]).applyMatrix4(C,a),o.normal.toArray(m,x),m[x+3]=o.constant}l.value=m,l.needsUpdate=!0}return e.numPlanes=_,e.numIntersection=0,m}}function ly(n){let e=new WeakMap;function t(o,a){return a===Ba?o.mapping=cs:a===Ha&&(o.mapping=ls),o}function i(o){if(o&&o.isTexture){const a=o.mapping;if(a===Ba||a===Ha)if(e.has(o)){const l=e.get(o).texture;return t(l,o.mapping)}else{const l=o.image;if(l&&l.height>0){const c=new M_(l.height);return c.fromEquirectangularTexture(n,o),e.set(o,c),o.addEventListener("dispose",s),t(c.texture,o.mapping)}else return null}}return o}function s(o){const a=o.target;a.removeEventListener("dispose",s);const l=e.get(a);l!==void 0&&(e.delete(a),l.dispose())}function r(){e=new WeakMap}return{get:i,dispose:r}}const es=4,Lu=[.125,.215,.35,.446,.526,.582],Si=20,da=new jc,Du=new Qe;let fa=null,pa=0,ma=0,ga=!1;const _i=(1+Math.sqrt(5))/2,Zi=1/_i,Iu=[new R(-_i,Zi,0),new R(_i,Zi,0),new R(-Zi,0,_i),new R(Zi,0,_i),new R(0,_i,-Zi),new R(0,_i,Zi),new R(-1,1,-1),new R(1,1,-1),new R(-1,1,1),new R(1,1,1)],uy=new R;class Nu{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._lodPlanes=[],this._sizeLods=[],this._sigmas=[],this._blurMaterial=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._compileMaterial(this._blurMaterial)}fromScene(e,t=0,i=.1,s=100,r={}){const{size:o=256,position:a=uy}=r;fa=this._renderer.getRenderTarget(),pa=this._renderer.getActiveCubeFace(),ma=this._renderer.getActiveMipmapLevel(),ga=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(o);const l=this._allocateTargets();return l.depthBuffer=!0,this._sceneToCubeUV(e,i,s,l,a),t>0&&this._blur(l,0,0,t),this._applyPMREM(l),this._cleanup(l),l}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Fu(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Ou(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose()}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodPlanes.length;e++)this._lodPlanes[e].dispose()}_cleanup(e){this._renderer.setRenderTarget(fa,pa,ma),this._renderer.xr.enabled=ga,e.scissorTest=!1,Br(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===cs||e.mapping===ls?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),fa=this._renderer.getRenderTarget(),pa=this._renderer.getActiveCubeFace(),ma=this._renderer.getActiveMipmapLevel(),ga=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const i=t||this._allocateTargets();return this._textureToCubeUV(e,i),this._applyPMREM(i),this._cleanup(i),i}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,i={magFilter:Tn,minFilter:Tn,generateMipmaps:!1,type:Ks,format:vn,colorSpace:us,depthBuffer:!1},s=Uu(e,t,i);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=Uu(e,t,i);const{_lodMax:r}=this;({sizeLods:this._sizeLods,lodPlanes:this._lodPlanes,sigmas:this._sigmas}=hy(r)),this._blurMaterial=dy(r,e,t)}return s}_compileMaterial(e){const t=new xt(this._lodPlanes[0],e);this._renderer.compile(t,da)}_sceneToCubeUV(e,t,i,s,r){const l=new sn(90,1,t,i),c=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],h=this._renderer,d=h.autoClear,f=h.toneMapping;h.getClearColor(Du),h.toneMapping=ri,h.autoClear=!1,h.state.buffers.depth.getReversed()&&(h.setRenderTarget(s),h.clearDepth(),h.setRenderTarget(null));const _=new qt({name:"PMREM.Background",side:Jt,depthWrite:!1,depthTest:!1}),m=new xt(new er,_);let p=!1;const C=e.background;C?C.isColor&&(_.color.copy(C),e.background=null,p=!0):(_.color.copy(Du),p=!0);for(let M=0;M<6;M++){const x=M%3;x===0?(l.up.set(0,c[M],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x+u[M],r.y,r.z)):x===1?(l.up.set(0,0,c[M]),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y+u[M],r.z)):(l.up.set(0,c[M],0),l.position.set(r.x,r.y,r.z),l.lookAt(r.x,r.y,r.z+u[M]));const P=this._cubeSize;Br(s,x*P,M>2?P:0,P,P),h.setRenderTarget(s),p&&h.render(m,l),h.render(e,l)}m.geometry.dispose(),m.material.dispose(),h.toneMapping=f,h.autoClear=d,e.background=C}_textureToCubeUV(e,t){const i=this._renderer,s=e.mapping===cs||e.mapping===ls;s?(this._cubemapMaterial===null&&(this._cubemapMaterial=Fu()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=Ou());const r=s?this._cubemapMaterial:this._equirectMaterial,o=new xt(this._lodPlanes[0],r),a=r.uniforms;a.envMap.value=e;const l=this._cubeSize;Br(t,0,0,3*l,2*l),i.setRenderTarget(t),i.render(o,da)}_applyPMREM(e){const t=this._renderer,i=t.autoClear;t.autoClear=!1;const s=this._lodPlanes.length;for(let r=1;r<s;r++){const o=Math.sqrt(this._sigmas[r]*this._sigmas[r]-this._sigmas[r-1]*this._sigmas[r-1]),a=Iu[(s-r-1)%Iu.length];this._blur(e,r-1,r,o,a)}t.autoClear=i}_blur(e,t,i,s,r){const o=this._pingPongRenderTarget;this._halfBlur(e,o,t,i,s,"latitudinal",r),this._halfBlur(o,e,i,i,s,"longitudinal",r)}_halfBlur(e,t,i,s,r,o,a){const l=this._renderer,c=this._blurMaterial;o!=="latitudinal"&&o!=="longitudinal"&&console.error("blur direction must be either latitudinal or longitudinal!");const u=3,h=new xt(this._lodPlanes[s],c),d=c.uniforms,f=this._sizeLods[i]-1,g=isFinite(r)?Math.PI/(2*f):2*Math.PI/(2*Si-1),_=r/g,m=isFinite(r)?1+Math.floor(u*_):Si;m>Si&&console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${m} samples when the maximum is set to ${Si}`);const p=[];let C=0;for(let T=0;T<Si;++T){const L=T/_,y=Math.exp(-L*L/2);p.push(y),T===0?C+=y:T<m&&(C+=2*y)}for(let T=0;T<p.length;T++)p[T]=p[T]/C;d.envMap.value=e.texture,d.samples.value=m,d.weights.value=p,d.latitudinal.value=o==="latitudinal",a&&(d.poleAxis.value=a);const{_lodMax:M}=this;d.dTheta.value=g,d.mipInt.value=M-i;const x=this._sizeLods[s],P=3*x*(s>M-es?s-M+es:0),w=4*(this._cubeSize-x);Br(t,P,w,3*x,2*x),l.setRenderTarget(t),l.render(h,da)}}function hy(n){const e=[],t=[],i=[];let s=n;const r=n-es+1+Lu.length;for(let o=0;o<r;o++){const a=Math.pow(2,s);t.push(a);let l=1/a;o>n-es?l=Lu[o-n+es-1]:o===0&&(l=0),i.push(l);const c=1/(a-2),u=-c,h=1+c,d=[u,u,h,u,h,h,u,u,h,h,u,h],f=6,g=6,_=3,m=2,p=1,C=new Float32Array(_*g*f),M=new Float32Array(m*g*f),x=new Float32Array(p*g*f);for(let w=0;w<f;w++){const T=w%3*2/3-1,L=w>2?0:-1,y=[T,L,0,T+2/3,L,0,T+2/3,L+1,0,T,L,0,T+2/3,L+1,0,T,L+1,0];C.set(y,_*g*w),M.set(d,m*g*w);const b=[w,w,w,w,w,w];x.set(b,p*g*w)}const P=new it;P.setAttribute("position",new hn(C,_)),P.setAttribute("uv",new hn(M,m)),P.setAttribute("faceIndex",new hn(x,p)),e.push(P),s>es&&s--}return{lodPlanes:e,sizeLods:t,sigmas:i}}function Uu(n,e,t){const i=new Ai(n,e,t);return i.texture.mapping=vo,i.texture.name="PMREM.cubeUv",i.scissorTest=!0,i}function Br(n,e,t,i,s){n.viewport.set(e,t,i,s),n.scissor.set(e,t,i,s)}function dy(n,e,t){const i=new Float32Array(Si),s=new R(0,1,0);return new ai({name:"SphericalGaussianBlur",defines:{n:Si,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${n}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:i},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:s}},vertexShader:Zc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:si,depthTest:!1,depthWrite:!1})}function Ou(){return new ai({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:Zc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:si,depthTest:!1,depthWrite:!1})}function Fu(){return new ai({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:Zc(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:si,depthTest:!1,depthWrite:!1})}function Zc(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}function fy(n){let e=new WeakMap,t=null;function i(a){if(a&&a.isTexture){const l=a.mapping,c=l===Ba||l===Ha,u=l===cs||l===ls;if(c||u){let h=e.get(a);const d=h!==void 0?h.texture.pmremVersion:0;if(a.isRenderTargetTexture&&a.pmremVersion!==d)return t===null&&(t=new Nu(n)),h=c?t.fromEquirectangular(a,h):t.fromCubemap(a,h),h.texture.pmremVersion=a.pmremVersion,e.set(a,h),h.texture;if(h!==void 0)return h.texture;{const f=a.image;return c&&f&&f.height>0||u&&f&&s(f)?(t===null&&(t=new Nu(n)),h=c?t.fromEquirectangular(a):t.fromCubemap(a),h.texture.pmremVersion=a.pmremVersion,e.set(a,h),a.addEventListener("dispose",r),h.texture):null}}}return a}function s(a){let l=0;const c=6;for(let u=0;u<c;u++)a[u]!==void 0&&l++;return l===c}function r(a){const l=a.target;l.removeEventListener("dispose",r);const c=e.get(l);c!==void 0&&(e.delete(l),c.dispose())}function o(){e=new WeakMap,t!==null&&(t.dispose(),t=null)}return{get:i,dispose:o}}function py(n){const e={};function t(i){if(e[i]!==void 0)return e[i];let s;switch(i){case"WEBGL_depth_texture":s=n.getExtension("WEBGL_depth_texture")||n.getExtension("MOZ_WEBGL_depth_texture")||n.getExtension("WEBKIT_WEBGL_depth_texture");break;case"EXT_texture_filter_anisotropic":s=n.getExtension("EXT_texture_filter_anisotropic")||n.getExtension("MOZ_EXT_texture_filter_anisotropic")||n.getExtension("WEBKIT_EXT_texture_filter_anisotropic");break;case"WEBGL_compressed_texture_s3tc":s=n.getExtension("WEBGL_compressed_texture_s3tc")||n.getExtension("MOZ_WEBGL_compressed_texture_s3tc")||n.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");break;case"WEBGL_compressed_texture_pvrtc":s=n.getExtension("WEBGL_compressed_texture_pvrtc")||n.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");break;default:s=n.getExtension(i)}return e[i]=s,s}return{has:function(i){return t(i)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(i){const s=t(i);return s===null&&Xs("THREE.WebGLRenderer: "+i+" extension not supported."),s}}}function my(n,e,t,i){const s={},r=new WeakMap;function o(h){const d=h.target;d.index!==null&&e.remove(d.index);for(const g in d.attributes)e.remove(d.attributes[g]);d.removeEventListener("dispose",o),delete s[d.id];const f=r.get(d);f&&(e.remove(f),r.delete(d)),i.releaseStatesOfGeometry(d),d.isInstancedBufferGeometry===!0&&delete d._maxInstanceCount,t.memory.geometries--}function a(h,d){return s[d.id]===!0||(d.addEventListener("dispose",o),s[d.id]=!0,t.memory.geometries++),d}function l(h){const d=h.attributes;for(const f in d)e.update(d[f],n.ARRAY_BUFFER)}function c(h){const d=[],f=h.index,g=h.attributes.position;let _=0;if(f!==null){const C=f.array;_=f.version;for(let M=0,x=C.length;M<x;M+=3){const P=C[M+0],w=C[M+1],T=C[M+2];d.push(P,w,w,T,T,P)}}else if(g!==void 0){const C=g.array;_=g.version;for(let M=0,x=C.length/3-1;M<x;M+=3){const P=M+0,w=M+1,T=M+2;d.push(P,w,w,T,T,P)}}else return;const m=new(Kh(d)?nd:td)(d,1);m.version=_;const p=r.get(h);p&&e.remove(p),r.set(h,m)}function u(h){const d=r.get(h);if(d){const f=h.index;f!==null&&d.version<f.version&&c(h)}else c(h);return r.get(h)}return{get:a,update:l,getWireframeAttribute:u}}function gy(n,e,t){let i;function s(d){i=d}let r,o;function a(d){r=d.type,o=d.bytesPerElement}function l(d,f){n.drawElements(i,f,r,d*o),t.update(f,i,1)}function c(d,f,g){g!==0&&(n.drawElementsInstanced(i,f,r,d*o,g),t.update(f,i,g))}function u(d,f,g){if(g===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(i,f,0,r,d,0,g);let m=0;for(let p=0;p<g;p++)m+=f[p];t.update(m,i,1)}function h(d,f,g,_){if(g===0)return;const m=e.get("WEBGL_multi_draw");if(m===null)for(let p=0;p<d.length;p++)c(d[p]/o,f[p],_[p]);else{m.multiDrawElementsInstancedWEBGL(i,f,0,r,d,0,_,0,g);let p=0;for(let C=0;C<g;C++)p+=f[C]*_[C];t.update(p,i,1)}}this.setMode=s,this.setIndex=a,this.render=l,this.renderInstances=c,this.renderMultiDraw=u,this.renderMultiDrawInstances=h}function _y(n){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function i(r,o,a){switch(t.calls++,o){case n.TRIANGLES:t.triangles+=a*(r/3);break;case n.LINES:t.lines+=a*(r/2);break;case n.LINE_STRIP:t.lines+=a*(r-1);break;case n.LINE_LOOP:t.lines+=a*r;break;case n.POINTS:t.points+=a*r;break;default:console.error("THREE.WebGLInfo: Unknown draw mode:",o);break}}function s(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:s,update:i}}function vy(n,e,t){const i=new WeakMap,s=new ct;function r(o,a,l){const c=o.morphTargetInfluences,u=a.morphAttributes.position||a.morphAttributes.normal||a.morphAttributes.color,h=u!==void 0?u.length:0;let d=i.get(a);if(d===void 0||d.count!==h){let b=function(){L.dispose(),i.delete(a),a.removeEventListener("dispose",b)};var f=b;d!==void 0&&d.texture.dispose();const g=a.morphAttributes.position!==void 0,_=a.morphAttributes.normal!==void 0,m=a.morphAttributes.color!==void 0,p=a.morphAttributes.position||[],C=a.morphAttributes.normal||[],M=a.morphAttributes.color||[];let x=0;g===!0&&(x=1),_===!0&&(x=2),m===!0&&(x=3);let P=a.attributes.position.count*x,w=1;P>e.maxTextureSize&&(w=Math.ceil(P/e.maxTextureSize),P=e.maxTextureSize);const T=new Float32Array(P*w*4*h),L=new Qh(T,P,w,h);L.type=Wn,L.needsUpdate=!0;const y=x*4;for(let N=0;N<h;N++){const O=p[N],G=C[N],V=M[N],X=P*w*4*N;for(let q=0;q<O.count;q++){const J=q*y;g===!0&&(s.fromBufferAttribute(O,q),T[X+J+0]=s.x,T[X+J+1]=s.y,T[X+J+2]=s.z,T[X+J+3]=0),_===!0&&(s.fromBufferAttribute(G,q),T[X+J+4]=s.x,T[X+J+5]=s.y,T[X+J+6]=s.z,T[X+J+7]=0),m===!0&&(s.fromBufferAttribute(V,q),T[X+J+8]=s.x,T[X+J+9]=s.y,T[X+J+10]=s.z,T[X+J+11]=V.itemSize===4?s.w:1)}}d={count:h,texture:L,size:new ue(P,w)},i.set(a,d),a.addEventListener("dispose",b)}if(o.isInstancedMesh===!0&&o.morphTexture!==null)l.getUniforms().setValue(n,"morphTexture",o.morphTexture,t);else{let g=0;for(let m=0;m<c.length;m++)g+=c[m];const _=a.morphTargetsRelative?1:1-g;l.getUniforms().setValue(n,"morphTargetBaseInfluence",_),l.getUniforms().setValue(n,"morphTargetInfluences",c)}l.getUniforms().setValue(n,"morphTargetsTexture",d.texture,t),l.getUniforms().setValue(n,"morphTargetsTextureSize",d.size)}return{update:r}}function xy(n,e,t,i){let s=new WeakMap;function r(l){const c=i.render.frame,u=l.geometry,h=e.get(l,u);if(s.get(h)!==c&&(e.update(h),s.set(h,c)),l.isInstancedMesh&&(l.hasEventListener("dispose",a)===!1&&l.addEventListener("dispose",a),s.get(l)!==c&&(t.update(l.instanceMatrix,n.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,n.ARRAY_BUFFER),s.set(l,c))),l.isSkinnedMesh){const d=l.skeleton;s.get(d)!==c&&(d.update(),s.set(d,c))}return h}function o(){s=new WeakMap}function a(l){const c=l.target;c.removeEventListener("dispose",a),t.remove(c.instanceMatrix),c.instanceColor!==null&&t.remove(c.instanceColor)}return{update:r,dispose:o}}const bd=new Vt,ku=new ld(1,1),Md=new Qh,wd=new o_,Ed=new rd,zu=[],Bu=[],Hu=new Float32Array(16),Gu=new Float32Array(9),Vu=new Float32Array(4);function gs(n,e,t){const i=n[0];if(i<=0||i>0)return n;const s=e*t;let r=zu[s];if(r===void 0&&(r=new Float32Array(s),zu[s]=r),e!==0){i.toArray(r,0);for(let o=1,a=0;o!==e;++o)a+=t,n[o].toArray(r,a)}return r}function Dt(n,e){if(n.length!==e.length)return!1;for(let t=0,i=n.length;t<i;t++)if(n[t]!==e[t])return!1;return!0}function It(n,e){for(let t=0,i=e.length;t<i;t++)n[t]=e[t]}function wo(n,e){let t=Bu[e];t===void 0&&(t=new Int32Array(e),Bu[e]=t);for(let i=0;i!==e;++i)t[i]=n.allocateTextureUnit();return t}function yy(n,e){const t=this.cache;t[0]!==e&&(n.uniform1f(this.addr,e),t[0]=e)}function Sy(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Dt(t,e))return;n.uniform2fv(this.addr,e),It(t,e)}}function by(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(n.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Dt(t,e))return;n.uniform3fv(this.addr,e),It(t,e)}}function My(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Dt(t,e))return;n.uniform4fv(this.addr,e),It(t,e)}}function wy(n,e){const t=this.cache,i=e.elements;if(i===void 0){if(Dt(t,e))return;n.uniformMatrix2fv(this.addr,!1,e),It(t,e)}else{if(Dt(t,i))return;Vu.set(i),n.uniformMatrix2fv(this.addr,!1,Vu),It(t,i)}}function Ey(n,e){const t=this.cache,i=e.elements;if(i===void 0){if(Dt(t,e))return;n.uniformMatrix3fv(this.addr,!1,e),It(t,e)}else{if(Dt(t,i))return;Gu.set(i),n.uniformMatrix3fv(this.addr,!1,Gu),It(t,i)}}function Ty(n,e){const t=this.cache,i=e.elements;if(i===void 0){if(Dt(t,e))return;n.uniformMatrix4fv(this.addr,!1,e),It(t,e)}else{if(Dt(t,i))return;Hu.set(i),n.uniformMatrix4fv(this.addr,!1,Hu),It(t,i)}}function Ay(n,e){const t=this.cache;t[0]!==e&&(n.uniform1i(this.addr,e),t[0]=e)}function Cy(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Dt(t,e))return;n.uniform2iv(this.addr,e),It(t,e)}}function Ry(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Dt(t,e))return;n.uniform3iv(this.addr,e),It(t,e)}}function Py(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Dt(t,e))return;n.uniform4iv(this.addr,e),It(t,e)}}function Ly(n,e){const t=this.cache;t[0]!==e&&(n.uniform1ui(this.addr,e),t[0]=e)}function Dy(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(n.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Dt(t,e))return;n.uniform2uiv(this.addr,e),It(t,e)}}function Iy(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(n.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Dt(t,e))return;n.uniform3uiv(this.addr,e),It(t,e)}}function Ny(n,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(n.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Dt(t,e))return;n.uniform4uiv(this.addr,e),It(t,e)}}function Uy(n,e,t){const i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s);let r;this.type===n.SAMPLER_2D_SHADOW?(ku.compareFunction=Jh,r=ku):r=bd,t.setTexture2D(e||r,s)}function Oy(n,e,t){const i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTexture3D(e||wd,s)}function Fy(n,e,t){const i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTextureCube(e||Ed,s)}function ky(n,e,t){const i=this.cache,s=t.allocateTextureUnit();i[0]!==s&&(n.uniform1i(this.addr,s),i[0]=s),t.setTexture2DArray(e||Md,s)}function zy(n){switch(n){case 5126:return yy;case 35664:return Sy;case 35665:return by;case 35666:return My;case 35674:return wy;case 35675:return Ey;case 35676:return Ty;case 5124:case 35670:return Ay;case 35667:case 35671:return Cy;case 35668:case 35672:return Ry;case 35669:case 35673:return Py;case 5125:return Ly;case 36294:return Dy;case 36295:return Iy;case 36296:return Ny;case 35678:case 36198:case 36298:case 36306:case 35682:return Uy;case 35679:case 36299:case 36307:return Oy;case 35680:case 36300:case 36308:case 36293:return Fy;case 36289:case 36303:case 36311:case 36292:return ky}}function By(n,e){n.uniform1fv(this.addr,e)}function Hy(n,e){const t=gs(e,this.size,2);n.uniform2fv(this.addr,t)}function Gy(n,e){const t=gs(e,this.size,3);n.uniform3fv(this.addr,t)}function Vy(n,e){const t=gs(e,this.size,4);n.uniform4fv(this.addr,t)}function Wy(n,e){const t=gs(e,this.size,4);n.uniformMatrix2fv(this.addr,!1,t)}function Xy(n,e){const t=gs(e,this.size,9);n.uniformMatrix3fv(this.addr,!1,t)}function $y(n,e){const t=gs(e,this.size,16);n.uniformMatrix4fv(this.addr,!1,t)}function Yy(n,e){n.uniform1iv(this.addr,e)}function qy(n,e){n.uniform2iv(this.addr,e)}function jy(n,e){n.uniform3iv(this.addr,e)}function Zy(n,e){n.uniform4iv(this.addr,e)}function Jy(n,e){n.uniform1uiv(this.addr,e)}function Ky(n,e){n.uniform2uiv(this.addr,e)}function Qy(n,e){n.uniform3uiv(this.addr,e)}function eS(n,e){n.uniform4uiv(this.addr,e)}function tS(n,e,t){const i=this.cache,s=e.length,r=wo(t,s);Dt(i,r)||(n.uniform1iv(this.addr,r),It(i,r));for(let o=0;o!==s;++o)t.setTexture2D(e[o]||bd,r[o])}function nS(n,e,t){const i=this.cache,s=e.length,r=wo(t,s);Dt(i,r)||(n.uniform1iv(this.addr,r),It(i,r));for(let o=0;o!==s;++o)t.setTexture3D(e[o]||wd,r[o])}function iS(n,e,t){const i=this.cache,s=e.length,r=wo(t,s);Dt(i,r)||(n.uniform1iv(this.addr,r),It(i,r));for(let o=0;o!==s;++o)t.setTextureCube(e[o]||Ed,r[o])}function sS(n,e,t){const i=this.cache,s=e.length,r=wo(t,s);Dt(i,r)||(n.uniform1iv(this.addr,r),It(i,r));for(let o=0;o!==s;++o)t.setTexture2DArray(e[o]||Md,r[o])}function rS(n){switch(n){case 5126:return By;case 35664:return Hy;case 35665:return Gy;case 35666:return Vy;case 35674:return Wy;case 35675:return Xy;case 35676:return $y;case 5124:case 35670:return Yy;case 35667:case 35671:return qy;case 35668:case 35672:return jy;case 35669:case 35673:return Zy;case 5125:return Jy;case 36294:return Ky;case 36295:return Qy;case 36296:return eS;case 35678:case 36198:case 36298:case 36306:case 35682:return tS;case 35679:case 36299:case 36307:return nS;case 35680:case 36300:case 36308:case 36293:return iS;case 36289:case 36303:case 36311:case 36292:return sS}}class oS{constructor(e,t,i){this.id=e,this.addr=i,this.cache=[],this.type=t.type,this.setValue=zy(t.type)}}class aS{constructor(e,t,i){this.id=e,this.addr=i,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=rS(t.type)}}class cS{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,i){const s=this.seq;for(let r=0,o=s.length;r!==o;++r){const a=s[r];a.setValue(e,t[a.id],i)}}}const _a=/(\w+)(\])?(\[|\.)?/g;function Wu(n,e){n.seq.push(e),n.map[e.id]=e}function lS(n,e,t){const i=n.name,s=i.length;for(_a.lastIndex=0;;){const r=_a.exec(i),o=_a.lastIndex;let a=r[1];const l=r[2]==="]",c=r[3];if(l&&(a=a|0),c===void 0||c==="["&&o+2===s){Wu(t,c===void 0?new oS(a,n,e):new aS(a,n,e));break}else{let h=t.map[a];h===void 0&&(h=new cS(a),Wu(t,h)),t=h}}}class Qr{constructor(e,t){this.seq=[],this.map={};const i=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let s=0;s<i;++s){const r=e.getActiveUniform(t,s),o=e.getUniformLocation(t,r.name);lS(r,o,this)}}setValue(e,t,i,s){const r=this.map[t];r!==void 0&&r.setValue(e,i,s)}setOptional(e,t,i){const s=t[i];s!==void 0&&this.setValue(e,i,s)}static upload(e,t,i,s){for(let r=0,o=t.length;r!==o;++r){const a=t[r],l=i[a.id];l.needsUpdate!==!1&&a.setValue(e,l.value,s)}}static seqWithValue(e,t){const i=[];for(let s=0,r=e.length;s!==r;++s){const o=e[s];o.id in t&&i.push(o)}return i}}function Xu(n,e,t){const i=n.createShader(e);return n.shaderSource(i,t),n.compileShader(i),i}const uS=37297;let hS=0;function dS(n,e){const t=n.split(`
`),i=[],s=Math.max(e-6,0),r=Math.min(e+6,t.length);for(let o=s;o<r;o++){const a=o+1;i.push(`${a===e?">":" "} ${a}: ${t[o]}`)}return i.join(`
`)}const $u=new Ve;function fS(n){tt._getMatrix($u,tt.workingColorSpace,n);const e=`mat3( ${$u.elements.map(t=>t.toFixed(4))} )`;switch(tt.getTransfer(n)){case oo:return[e,"LinearTransferOETF"];case rt:return[e,"sRGBTransferOETF"];default:return console.warn("THREE.WebGLProgram: Unsupported color space: ",n),[e,"LinearTransferOETF"]}}function Yu(n,e,t){const i=n.getShaderParameter(e,n.COMPILE_STATUS),r=(n.getShaderInfoLog(e)||"").trim();if(i&&r==="")return"";const o=/ERROR: 0:(\d+)/.exec(r);if(o){const a=parseInt(o[1]);return t.toUpperCase()+`

`+r+`

`+dS(n.getShaderSource(e),a)}else return r}function pS(n,e){const t=fS(e);return[`vec4 ${n}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}function mS(n,e){let t;switch(e){case Dg:t="Linear";break;case Ig:t="Reinhard";break;case Ng:t="Cineon";break;case Ug:t="ACESFilmic";break;case Fg:t="AgX";break;case kg:t="Neutral";break;case Og:t="Custom";break;default:console.warn("THREE.WebGLProgram: Unsupported toneMapping:",e),t="Linear"}return"vec3 "+n+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const Hr=new R;function gS(){tt.getLuminanceCoefficients(Hr);const n=Hr.x.toFixed(4),e=Hr.y.toFixed(4),t=Hr.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${n}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function _S(n){return[n.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",n.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Ds).join(`
`)}function vS(n){const e=[];for(const t in n){const i=n[t];i!==!1&&e.push("#define "+t+" "+i)}return e.join(`
`)}function xS(n,e){const t={},i=n.getProgramParameter(e,n.ACTIVE_ATTRIBUTES);for(let s=0;s<i;s++){const r=n.getActiveAttrib(e,s),o=r.name;let a=1;r.type===n.FLOAT_MAT2&&(a=2),r.type===n.FLOAT_MAT3&&(a=3),r.type===n.FLOAT_MAT4&&(a=4),t[o]={type:r.type,location:n.getAttribLocation(e,o),locationSize:a}}return t}function Ds(n){return n!==""}function qu(n,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return n.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function ju(n,e){return n.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const yS=/^[ \t]*#include +<([\w\d./]+)>/gm;function Mc(n){return n.replace(yS,bS)}const SS=new Map;function bS(n,e){let t=$e[e];if(t===void 0){const i=SS.get(e);if(i!==void 0)t=$e[i],console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,i);else throw new Error("Can not resolve #include <"+e+">")}return Mc(t)}const MS=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Zu(n){return n.replace(MS,wS)}function wS(n,e,t,i){let s="";for(let r=parseInt(e);r<parseInt(t);r++)s+=i.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function Ju(n){let e=`precision ${n.precision} float;
	precision ${n.precision} int;
	precision ${n.precision} sampler2D;
	precision ${n.precision} samplerCube;
	precision ${n.precision} sampler3D;
	precision ${n.precision} sampler2DArray;
	precision ${n.precision} sampler2DShadow;
	precision ${n.precision} samplerCubeShadow;
	precision ${n.precision} sampler2DArrayShadow;
	precision ${n.precision} isampler2D;
	precision ${n.precision} isampler3D;
	precision ${n.precision} isamplerCube;
	precision ${n.precision} isampler2DArray;
	precision ${n.precision} usampler2D;
	precision ${n.precision} usampler3D;
	precision ${n.precision} usamplerCube;
	precision ${n.precision} usampler2DArray;
	`;return n.precision==="highp"?e+=`
#define HIGH_PRECISION`:n.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:n.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}function ES(n){let e="SHADOWMAP_TYPE_BASIC";return n.shadowMapType===Bh?e="SHADOWMAP_TYPE_PCF":n.shadowMapType===hg?e="SHADOWMAP_TYPE_PCF_SOFT":n.shadowMapType===Bn&&(e="SHADOWMAP_TYPE_VSM"),e}function TS(n){let e="ENVMAP_TYPE_CUBE";if(n.envMap)switch(n.envMapMode){case cs:case ls:e="ENVMAP_TYPE_CUBE";break;case vo:e="ENVMAP_TYPE_CUBE_UV";break}return e}function AS(n){let e="ENVMAP_MODE_REFLECTION";return n.envMap&&n.envMapMode===ls&&(e="ENVMAP_MODE_REFRACTION"),e}function CS(n){let e="ENVMAP_BLENDING_NONE";if(n.envMap)switch(n.combine){case Hh:e="ENVMAP_BLENDING_MULTIPLY";break;case Pg:e="ENVMAP_BLENDING_MIX";break;case Lg:e="ENVMAP_BLENDING_ADD";break}return e}function RS(n){const e=n.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,i=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:i,maxMip:t}}function PS(n,e,t,i){const s=n.getContext(),r=t.defines;let o=t.vertexShader,a=t.fragmentShader;const l=ES(t),c=TS(t),u=AS(t),h=CS(t),d=RS(t),f=_S(t),g=vS(r),_=s.createProgram();let m,p,C=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(Ds).join(`
`),m.length>0&&(m+=`
`),p=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g].filter(Ds).join(`
`),p.length>0&&(p+=`
`)):(m=[Ju(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Ds).join(`
`),p=[Ju(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,g,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+c:"",t.envMap?"#define "+u:"",t.envMap?"#define "+h:"",d?"#define CUBEUV_TEXEL_WIDTH "+d.texelWidth:"",d?"#define CUBEUV_TEXEL_HEIGHT "+d.texelHeight:"",d?"#define CUBEUV_MAX_MIP "+d.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor||t.batchingColor?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+l:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==ri?"#define TONE_MAPPING":"",t.toneMapping!==ri?$e.tonemapping_pars_fragment:"",t.toneMapping!==ri?mS("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",$e.colorspace_pars_fragment,pS("linearToOutputTexel",t.outputColorSpace),gS(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(Ds).join(`
`)),o=Mc(o),o=qu(o,t),o=ju(o,t),a=Mc(a),a=qu(a,t),a=ju(a,t),o=Zu(o),a=Zu(a),t.isRawShaderMaterial!==!0&&(C=`#version 300 es
`,m=[f,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+m,p=["#define varying in",t.glslVersion===$l?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===$l?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+p);const M=C+m+o,x=C+p+a,P=Xu(s,s.VERTEX_SHADER,M),w=Xu(s,s.FRAGMENT_SHADER,x);s.attachShader(_,P),s.attachShader(_,w),t.index0AttributeName!==void 0?s.bindAttribLocation(_,0,t.index0AttributeName):t.morphTargets===!0&&s.bindAttribLocation(_,0,"position"),s.linkProgram(_);function T(N){if(n.debug.checkShaderErrors){const O=s.getProgramInfoLog(_)||"",G=s.getShaderInfoLog(P)||"",V=s.getShaderInfoLog(w)||"",X=O.trim(),q=G.trim(),J=V.trim();let z=!0,H=!0;if(s.getProgramParameter(_,s.LINK_STATUS)===!1)if(z=!1,typeof n.debug.onShaderError=="function")n.debug.onShaderError(s,_,P,w);else{const D=Yu(s,P,"vertex"),W=Yu(s,w,"fragment");console.error("THREE.WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(_,s.VALIDATE_STATUS)+`

Material Name: `+N.name+`
Material Type: `+N.type+`

Program Info Log: `+X+`
`+D+`
`+W)}else X!==""?console.warn("THREE.WebGLProgram: Program Info Log:",X):(q===""||J==="")&&(H=!1);H&&(N.diagnostics={runnable:z,programLog:X,vertexShader:{log:q,prefix:m},fragmentShader:{log:J,prefix:p}})}s.deleteShader(P),s.deleteShader(w),L=new Qr(s,_),y=xS(s,_)}let L;this.getUniforms=function(){return L===void 0&&T(this),L};let y;this.getAttributes=function(){return y===void 0&&T(this),y};let b=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return b===!1&&(b=s.getProgramParameter(_,uS)),b},this.destroy=function(){i.releaseStatesOfProgram(this),s.deleteProgram(_),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=hS++,this.cacheKey=e,this.usedTimes=1,this.program=_,this.vertexShader=P,this.fragmentShader=w,this}let LS=0;class DS{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,i=e.fragmentShader,s=this._getShaderStage(t),r=this._getShaderStage(i),o=this._getShaderCacheForMaterial(e);return o.has(s)===!1&&(o.add(s),s.usedTimes++),o.has(r)===!1&&(o.add(r),r.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const i of t)i.usedTimes--,i.usedTimes===0&&this.shaderCache.delete(i.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let i=t.get(e);return i===void 0&&(i=new Set,t.set(e,i)),i}_getShaderStage(e){const t=this.shaderCache;let i=t.get(e);return i===void 0&&(i=new IS(e),t.set(e,i)),i}}class IS{constructor(e){this.id=LS++,this.code=e,this.usedTimes=0}}function NS(n,e,t,i,s,r,o){const a=new Hc,l=new DS,c=new Set,u=[],h=s.logarithmicDepthBuffer,d=s.vertexTextures;let f=s.precision;const g={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distanceRGBA",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function _(y){return c.add(y),y===0?"uv":`uv${y}`}function m(y,b,N,O,G){const V=O.fog,X=G.geometry,q=y.isMeshStandardMaterial?O.environment:null,J=(y.isMeshStandardMaterial?t:e).get(y.envMap||q),z=J&&J.mapping===vo?J.image.height:null,H=g[y.type];y.precision!==null&&(f=s.getMaxPrecision(y.precision),f!==y.precision&&console.warn("THREE.WebGLProgram.getParameters:",y.precision,"not supported, using",f,"instead."));const D=X.morphAttributes.position||X.morphAttributes.normal||X.morphAttributes.color,W=D!==void 0?D.length:0;let Q=0;X.morphAttributes.position!==void 0&&(Q=1),X.morphAttributes.normal!==void 0&&(Q=2),X.morphAttributes.color!==void 0&&(Q=3);let me,De,Ie,j;if(H){const nt=Mn[H];me=nt.vertexShader,De=nt.fragmentShader}else me=y.vertexShader,De=y.fragmentShader,l.update(y),Ie=l.getVertexShaderID(y),j=l.getFragmentShaderID(y);const te=n.getRenderTarget(),xe=n.state.buffers.depth.getReversed(),Fe=G.isInstancedMesh===!0,Te=G.isBatchedMesh===!0,Je=!!y.map,kt=!!y.matcap,I=!!J,_t=!!y.aoMap,He=!!y.lightMap,ke=!!y.bumpMap,Me=!!y.normalMap,vt=!!y.displacementMap,we=!!y.emissiveMap,Xe=!!y.metalnessMap,Nt=!!y.roughnessMap,wt=y.anisotropy>0,A=y.clearcoat>0,v=y.dispersion>0,B=y.iridescence>0,K=y.sheen>0,ne=y.transmission>0,Z=wt&&!!y.anisotropyMap,Re=A&&!!y.clearcoatMap,ce=A&&!!y.clearcoatNormalMap,Ee=A&&!!y.clearcoatRoughnessMap,Ae=B&&!!y.iridescenceMap,re=B&&!!y.iridescenceThicknessMap,ge=K&&!!y.sheenColorMap,Oe=K&&!!y.sheenRoughnessMap,Ce=!!y.specularMap,fe=!!y.specularColorMap,Ge=!!y.specularIntensityMap,U=ne&&!!y.transmissionMap,oe=ne&&!!y.thicknessMap,he=!!y.gradientMap,ye=!!y.alphaMap,ie=y.alphaTest>0,ee=!!y.alphaHash,be=!!y.extensions;let Be=ri;y.toneMapped&&(te===null||te.isXRRenderTarget===!0)&&(Be=n.toneMapping);const dt={shaderID:H,shaderType:y.type,shaderName:y.name,vertexShader:me,fragmentShader:De,defines:y.defines,customVertexShaderID:Ie,customFragmentShaderID:j,isRawShaderMaterial:y.isRawShaderMaterial===!0,glslVersion:y.glslVersion,precision:f,batching:Te,batchingColor:Te&&G._colorsTexture!==null,instancing:Fe,instancingColor:Fe&&G.instanceColor!==null,instancingMorph:Fe&&G.morphTexture!==null,supportsVertexTextures:d,outputColorSpace:te===null?n.outputColorSpace:te.isXRRenderTarget===!0?te.texture.colorSpace:us,alphaToCoverage:!!y.alphaToCoverage,map:Je,matcap:kt,envMap:I,envMapMode:I&&J.mapping,envMapCubeUVHeight:z,aoMap:_t,lightMap:He,bumpMap:ke,normalMap:Me,displacementMap:d&&vt,emissiveMap:we,normalMapObjectSpace:Me&&y.normalMapType===Vg,normalMapTangentSpace:Me&&y.normalMapType===Gg,metalnessMap:Xe,roughnessMap:Nt,anisotropy:wt,anisotropyMap:Z,clearcoat:A,clearcoatMap:Re,clearcoatNormalMap:ce,clearcoatRoughnessMap:Ee,dispersion:v,iridescence:B,iridescenceMap:Ae,iridescenceThicknessMap:re,sheen:K,sheenColorMap:ge,sheenRoughnessMap:Oe,specularMap:Ce,specularColorMap:fe,specularIntensityMap:Ge,transmission:ne,transmissionMap:U,thicknessMap:oe,gradientMap:he,opaque:y.transparent===!1&&y.blending===ns&&y.alphaToCoverage===!1,alphaMap:ye,alphaTest:ie,alphaHash:ee,combine:y.combine,mapUv:Je&&_(y.map.channel),aoMapUv:_t&&_(y.aoMap.channel),lightMapUv:He&&_(y.lightMap.channel),bumpMapUv:ke&&_(y.bumpMap.channel),normalMapUv:Me&&_(y.normalMap.channel),displacementMapUv:vt&&_(y.displacementMap.channel),emissiveMapUv:we&&_(y.emissiveMap.channel),metalnessMapUv:Xe&&_(y.metalnessMap.channel),roughnessMapUv:Nt&&_(y.roughnessMap.channel),anisotropyMapUv:Z&&_(y.anisotropyMap.channel),clearcoatMapUv:Re&&_(y.clearcoatMap.channel),clearcoatNormalMapUv:ce&&_(y.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:Ee&&_(y.clearcoatRoughnessMap.channel),iridescenceMapUv:Ae&&_(y.iridescenceMap.channel),iridescenceThicknessMapUv:re&&_(y.iridescenceThicknessMap.channel),sheenColorMapUv:ge&&_(y.sheenColorMap.channel),sheenRoughnessMapUv:Oe&&_(y.sheenRoughnessMap.channel),specularMapUv:Ce&&_(y.specularMap.channel),specularColorMapUv:fe&&_(y.specularColorMap.channel),specularIntensityMapUv:Ge&&_(y.specularIntensityMap.channel),transmissionMapUv:U&&_(y.transmissionMap.channel),thicknessMapUv:oe&&_(y.thicknessMap.channel),alphaMapUv:ye&&_(y.alphaMap.channel),vertexTangents:!!X.attributes.tangent&&(Me||wt),vertexColors:y.vertexColors,vertexAlphas:y.vertexColors===!0&&!!X.attributes.color&&X.attributes.color.itemSize===4,pointsUvs:G.isPoints===!0&&!!X.attributes.uv&&(Je||ye),fog:!!V,useFog:y.fog===!0,fogExp2:!!V&&V.isFogExp2,flatShading:y.flatShading===!0&&y.wireframe===!1,sizeAttenuation:y.sizeAttenuation===!0,logarithmicDepthBuffer:h,reversedDepthBuffer:xe,skinning:G.isSkinnedMesh===!0,morphTargets:X.morphAttributes.position!==void 0,morphNormals:X.morphAttributes.normal!==void 0,morphColors:X.morphAttributes.color!==void 0,morphTargetsCount:W,morphTextureStride:Q,numDirLights:b.directional.length,numPointLights:b.point.length,numSpotLights:b.spot.length,numSpotLightMaps:b.spotLightMap.length,numRectAreaLights:b.rectArea.length,numHemiLights:b.hemi.length,numDirLightShadows:b.directionalShadowMap.length,numPointLightShadows:b.pointShadowMap.length,numSpotLightShadows:b.spotShadowMap.length,numSpotLightShadowsWithMaps:b.numSpotLightShadowsWithMaps,numLightProbes:b.numLightProbes,numClippingPlanes:o.numPlanes,numClipIntersection:o.numIntersection,dithering:y.dithering,shadowMapEnabled:n.shadowMap.enabled&&N.length>0,shadowMapType:n.shadowMap.type,toneMapping:Be,decodeVideoTexture:Je&&y.map.isVideoTexture===!0&&tt.getTransfer(y.map.colorSpace)===rt,decodeVideoTextureEmissive:we&&y.emissiveMap.isVideoTexture===!0&&tt.getTransfer(y.emissiveMap.colorSpace)===rt,premultipliedAlpha:y.premultipliedAlpha,doubleSided:y.side===cn,flipSided:y.side===Jt,useDepthPacking:y.depthPacking>=0,depthPacking:y.depthPacking||0,index0AttributeName:y.index0AttributeName,extensionClipCullDistance:be&&y.extensions.clipCullDistance===!0&&i.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(be&&y.extensions.multiDraw===!0||Te)&&i.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:i.has("KHR_parallel_shader_compile"),customProgramCacheKey:y.customProgramCacheKey()};return dt.vertexUv1s=c.has(1),dt.vertexUv2s=c.has(2),dt.vertexUv3s=c.has(3),c.clear(),dt}function p(y){const b=[];if(y.shaderID?b.push(y.shaderID):(b.push(y.customVertexShaderID),b.push(y.customFragmentShaderID)),y.defines!==void 0)for(const N in y.defines)b.push(N),b.push(y.defines[N]);return y.isRawShaderMaterial===!1&&(C(b,y),M(b,y),b.push(n.outputColorSpace)),b.push(y.customProgramCacheKey),b.join()}function C(y,b){y.push(b.precision),y.push(b.outputColorSpace),y.push(b.envMapMode),y.push(b.envMapCubeUVHeight),y.push(b.mapUv),y.push(b.alphaMapUv),y.push(b.lightMapUv),y.push(b.aoMapUv),y.push(b.bumpMapUv),y.push(b.normalMapUv),y.push(b.displacementMapUv),y.push(b.emissiveMapUv),y.push(b.metalnessMapUv),y.push(b.roughnessMapUv),y.push(b.anisotropyMapUv),y.push(b.clearcoatMapUv),y.push(b.clearcoatNormalMapUv),y.push(b.clearcoatRoughnessMapUv),y.push(b.iridescenceMapUv),y.push(b.iridescenceThicknessMapUv),y.push(b.sheenColorMapUv),y.push(b.sheenRoughnessMapUv),y.push(b.specularMapUv),y.push(b.specularColorMapUv),y.push(b.specularIntensityMapUv),y.push(b.transmissionMapUv),y.push(b.thicknessMapUv),y.push(b.combine),y.push(b.fogExp2),y.push(b.sizeAttenuation),y.push(b.morphTargetsCount),y.push(b.morphAttributeCount),y.push(b.numDirLights),y.push(b.numPointLights),y.push(b.numSpotLights),y.push(b.numSpotLightMaps),y.push(b.numHemiLights),y.push(b.numRectAreaLights),y.push(b.numDirLightShadows),y.push(b.numPointLightShadows),y.push(b.numSpotLightShadows),y.push(b.numSpotLightShadowsWithMaps),y.push(b.numLightProbes),y.push(b.shadowMapType),y.push(b.toneMapping),y.push(b.numClippingPlanes),y.push(b.numClipIntersection),y.push(b.depthPacking)}function M(y,b){a.disableAll(),b.supportsVertexTextures&&a.enable(0),b.instancing&&a.enable(1),b.instancingColor&&a.enable(2),b.instancingMorph&&a.enable(3),b.matcap&&a.enable(4),b.envMap&&a.enable(5),b.normalMapObjectSpace&&a.enable(6),b.normalMapTangentSpace&&a.enable(7),b.clearcoat&&a.enable(8),b.iridescence&&a.enable(9),b.alphaTest&&a.enable(10),b.vertexColors&&a.enable(11),b.vertexAlphas&&a.enable(12),b.vertexUv1s&&a.enable(13),b.vertexUv2s&&a.enable(14),b.vertexUv3s&&a.enable(15),b.vertexTangents&&a.enable(16),b.anisotropy&&a.enable(17),b.alphaHash&&a.enable(18),b.batching&&a.enable(19),b.dispersion&&a.enable(20),b.batchingColor&&a.enable(21),b.gradientMap&&a.enable(22),y.push(a.mask),a.disableAll(),b.fog&&a.enable(0),b.useFog&&a.enable(1),b.flatShading&&a.enable(2),b.logarithmicDepthBuffer&&a.enable(3),b.reversedDepthBuffer&&a.enable(4),b.skinning&&a.enable(5),b.morphTargets&&a.enable(6),b.morphNormals&&a.enable(7),b.morphColors&&a.enable(8),b.premultipliedAlpha&&a.enable(9),b.shadowMapEnabled&&a.enable(10),b.doubleSided&&a.enable(11),b.flipSided&&a.enable(12),b.useDepthPacking&&a.enable(13),b.dithering&&a.enable(14),b.transmission&&a.enable(15),b.sheen&&a.enable(16),b.opaque&&a.enable(17),b.pointsUvs&&a.enable(18),b.decodeVideoTexture&&a.enable(19),b.decodeVideoTextureEmissive&&a.enable(20),b.alphaToCoverage&&a.enable(21),y.push(a.mask)}function x(y){const b=g[y.type];let N;if(b){const O=Mn[b];N=x_.clone(O.uniforms)}else N=y.uniforms;return N}function P(y,b){let N;for(let O=0,G=u.length;O<G;O++){const V=u[O];if(V.cacheKey===b){N=V,++N.usedTimes;break}}return N===void 0&&(N=new PS(n,b,y,r),u.push(N)),N}function w(y){if(--y.usedTimes===0){const b=u.indexOf(y);u[b]=u[u.length-1],u.pop(),y.destroy()}}function T(y){l.remove(y)}function L(){l.dispose()}return{getParameters:m,getProgramCacheKey:p,getUniforms:x,acquireProgram:P,releaseProgram:w,releaseShaderCache:T,programs:u,dispose:L}}function US(){let n=new WeakMap;function e(o){return n.has(o)}function t(o){let a=n.get(o);return a===void 0&&(a={},n.set(o,a)),a}function i(o){n.delete(o)}function s(o,a,l){n.get(o)[a]=l}function r(){n=new WeakMap}return{has:e,get:t,remove:i,update:s,dispose:r}}function OS(n,e){return n.groupOrder!==e.groupOrder?n.groupOrder-e.groupOrder:n.renderOrder!==e.renderOrder?n.renderOrder-e.renderOrder:n.material.id!==e.material.id?n.material.id-e.material.id:n.z!==e.z?n.z-e.z:n.id-e.id}function Ku(n,e){return n.groupOrder!==e.groupOrder?n.groupOrder-e.groupOrder:n.renderOrder!==e.renderOrder?n.renderOrder-e.renderOrder:n.z!==e.z?e.z-n.z:n.id-e.id}function Qu(){const n=[];let e=0;const t=[],i=[],s=[];function r(){e=0,t.length=0,i.length=0,s.length=0}function o(h,d,f,g,_,m){let p=n[e];return p===void 0?(p={id:h.id,object:h,geometry:d,material:f,groupOrder:g,renderOrder:h.renderOrder,z:_,group:m},n[e]=p):(p.id=h.id,p.object=h,p.geometry=d,p.material=f,p.groupOrder=g,p.renderOrder=h.renderOrder,p.z=_,p.group=m),e++,p}function a(h,d,f,g,_,m){const p=o(h,d,f,g,_,m);f.transmission>0?i.push(p):f.transparent===!0?s.push(p):t.push(p)}function l(h,d,f,g,_,m){const p=o(h,d,f,g,_,m);f.transmission>0?i.unshift(p):f.transparent===!0?s.unshift(p):t.unshift(p)}function c(h,d){t.length>1&&t.sort(h||OS),i.length>1&&i.sort(d||Ku),s.length>1&&s.sort(d||Ku)}function u(){for(let h=e,d=n.length;h<d;h++){const f=n[h];if(f.id===null)break;f.id=null,f.object=null,f.geometry=null,f.material=null,f.group=null}}return{opaque:t,transmissive:i,transparent:s,init:r,push:a,unshift:l,finish:u,sort:c}}function FS(){let n=new WeakMap;function e(i,s){const r=n.get(i);let o;return r===void 0?(o=new Qu,n.set(i,[o])):s>=r.length?(o=new Qu,r.push(o)):o=r[s],o}function t(){n=new WeakMap}return{get:e,dispose:t}}function kS(){const n={};return{get:function(e){if(n[e.id]!==void 0)return n[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new R,color:new Qe};break;case"SpotLight":t={position:new R,direction:new R,color:new Qe,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new R,color:new Qe,distance:0,decay:0};break;case"HemisphereLight":t={direction:new R,skyColor:new Qe,groundColor:new Qe};break;case"RectAreaLight":t={color:new Qe,position:new R,halfWidth:new R,halfHeight:new R};break}return n[e.id]=t,t}}}function zS(){const n={};return{get:function(e){if(n[e.id]!==void 0)return n[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ue};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ue};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ue,shadowCameraNear:1,shadowCameraFar:1e3};break}return n[e.id]=t,t}}}let BS=0;function HS(n,e){return(e.castShadow?2:0)-(n.castShadow?2:0)+(e.map?1:0)-(n.map?1:0)}function GS(n){const e=new kS,t=zS(),i={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let c=0;c<9;c++)i.probe.push(new R);const s=new R,r=new ft,o=new ft;function a(c){let u=0,h=0,d=0;for(let y=0;y<9;y++)i.probe[y].set(0,0,0);let f=0,g=0,_=0,m=0,p=0,C=0,M=0,x=0,P=0,w=0,T=0;c.sort(HS);for(let y=0,b=c.length;y<b;y++){const N=c[y],O=N.color,G=N.intensity,V=N.distance,X=N.shadow&&N.shadow.map?N.shadow.map.texture:null;if(N.isAmbientLight)u+=O.r*G,h+=O.g*G,d+=O.b*G;else if(N.isLightProbe){for(let q=0;q<9;q++)i.probe[q].addScaledVector(N.sh.coefficients[q],G);T++}else if(N.isDirectionalLight){const q=e.get(N);if(q.color.copy(N.color).multiplyScalar(N.intensity),N.castShadow){const J=N.shadow,z=t.get(N);z.shadowIntensity=J.intensity,z.shadowBias=J.bias,z.shadowNormalBias=J.normalBias,z.shadowRadius=J.radius,z.shadowMapSize=J.mapSize,i.directionalShadow[f]=z,i.directionalShadowMap[f]=X,i.directionalShadowMatrix[f]=N.shadow.matrix,C++}i.directional[f]=q,f++}else if(N.isSpotLight){const q=e.get(N);q.position.setFromMatrixPosition(N.matrixWorld),q.color.copy(O).multiplyScalar(G),q.distance=V,q.coneCos=Math.cos(N.angle),q.penumbraCos=Math.cos(N.angle*(1-N.penumbra)),q.decay=N.decay,i.spot[_]=q;const J=N.shadow;if(N.map&&(i.spotLightMap[P]=N.map,P++,J.updateMatrices(N),N.castShadow&&w++),i.spotLightMatrix[_]=J.matrix,N.castShadow){const z=t.get(N);z.shadowIntensity=J.intensity,z.shadowBias=J.bias,z.shadowNormalBias=J.normalBias,z.shadowRadius=J.radius,z.shadowMapSize=J.mapSize,i.spotShadow[_]=z,i.spotShadowMap[_]=X,x++}_++}else if(N.isRectAreaLight){const q=e.get(N);q.color.copy(O).multiplyScalar(G),q.halfWidth.set(N.width*.5,0,0),q.halfHeight.set(0,N.height*.5,0),i.rectArea[m]=q,m++}else if(N.isPointLight){const q=e.get(N);if(q.color.copy(N.color).multiplyScalar(N.intensity),q.distance=N.distance,q.decay=N.decay,N.castShadow){const J=N.shadow,z=t.get(N);z.shadowIntensity=J.intensity,z.shadowBias=J.bias,z.shadowNormalBias=J.normalBias,z.shadowRadius=J.radius,z.shadowMapSize=J.mapSize,z.shadowCameraNear=J.camera.near,z.shadowCameraFar=J.camera.far,i.pointShadow[g]=z,i.pointShadowMap[g]=X,i.pointShadowMatrix[g]=N.shadow.matrix,M++}i.point[g]=q,g++}else if(N.isHemisphereLight){const q=e.get(N);q.skyColor.copy(N.color).multiplyScalar(G),q.groundColor.copy(N.groundColor).multiplyScalar(G),i.hemi[p]=q,p++}}m>0&&(n.has("OES_texture_float_linear")===!0?(i.rectAreaLTC1=de.LTC_FLOAT_1,i.rectAreaLTC2=de.LTC_FLOAT_2):(i.rectAreaLTC1=de.LTC_HALF_1,i.rectAreaLTC2=de.LTC_HALF_2)),i.ambient[0]=u,i.ambient[1]=h,i.ambient[2]=d;const L=i.hash;(L.directionalLength!==f||L.pointLength!==g||L.spotLength!==_||L.rectAreaLength!==m||L.hemiLength!==p||L.numDirectionalShadows!==C||L.numPointShadows!==M||L.numSpotShadows!==x||L.numSpotMaps!==P||L.numLightProbes!==T)&&(i.directional.length=f,i.spot.length=_,i.rectArea.length=m,i.point.length=g,i.hemi.length=p,i.directionalShadow.length=C,i.directionalShadowMap.length=C,i.pointShadow.length=M,i.pointShadowMap.length=M,i.spotShadow.length=x,i.spotShadowMap.length=x,i.directionalShadowMatrix.length=C,i.pointShadowMatrix.length=M,i.spotLightMatrix.length=x+P-w,i.spotLightMap.length=P,i.numSpotLightShadowsWithMaps=w,i.numLightProbes=T,L.directionalLength=f,L.pointLength=g,L.spotLength=_,L.rectAreaLength=m,L.hemiLength=p,L.numDirectionalShadows=C,L.numPointShadows=M,L.numSpotShadows=x,L.numSpotMaps=P,L.numLightProbes=T,i.version=BS++)}function l(c,u){let h=0,d=0,f=0,g=0,_=0;const m=u.matrixWorldInverse;for(let p=0,C=c.length;p<C;p++){const M=c[p];if(M.isDirectionalLight){const x=i.directional[h];x.direction.setFromMatrixPosition(M.matrixWorld),s.setFromMatrixPosition(M.target.matrixWorld),x.direction.sub(s),x.direction.transformDirection(m),h++}else if(M.isSpotLight){const x=i.spot[f];x.position.setFromMatrixPosition(M.matrixWorld),x.position.applyMatrix4(m),x.direction.setFromMatrixPosition(M.matrixWorld),s.setFromMatrixPosition(M.target.matrixWorld),x.direction.sub(s),x.direction.transformDirection(m),f++}else if(M.isRectAreaLight){const x=i.rectArea[g];x.position.setFromMatrixPosition(M.matrixWorld),x.position.applyMatrix4(m),o.identity(),r.copy(M.matrixWorld),r.premultiply(m),o.extractRotation(r),x.halfWidth.set(M.width*.5,0,0),x.halfHeight.set(0,M.height*.5,0),x.halfWidth.applyMatrix4(o),x.halfHeight.applyMatrix4(o),g++}else if(M.isPointLight){const x=i.point[d];x.position.setFromMatrixPosition(M.matrixWorld),x.position.applyMatrix4(m),d++}else if(M.isHemisphereLight){const x=i.hemi[_];x.direction.setFromMatrixPosition(M.matrixWorld),x.direction.transformDirection(m),_++}}}return{setup:a,setupView:l,state:i}}function eh(n){const e=new GS(n),t=[],i=[];function s(u){c.camera=u,t.length=0,i.length=0}function r(u){t.push(u)}function o(u){i.push(u)}function a(){e.setup(t)}function l(u){e.setupView(t,u)}const c={lightsArray:t,shadowsArray:i,camera:null,lights:e,transmissionRenderTarget:{}};return{init:s,state:c,setupLights:a,setupLightsView:l,pushLight:r,pushShadow:o}}function VS(n){let e=new WeakMap;function t(s,r=0){const o=e.get(s);let a;return o===void 0?(a=new eh(n),e.set(s,[a])):r>=o.length?(a=new eh(n),o.push(a)):a=o[r],a}function i(){e=new WeakMap}return{get:t,dispose:i}}const WS=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,XS=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;function $S(n,e,t){let i=new Wc;const s=new ue,r=new ue,o=new ct,a=new av({depthPacking:Hg}),l=new cv,c={},u=t.maxTextureSize,h={[oi]:Jt,[Jt]:oi,[cn]:cn},d=new ai({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new ue},radius:{value:4}},vertexShader:WS,fragmentShader:XS}),f=d.clone();f.defines.HORIZONTAL_PASS=1;const g=new it;g.setAttribute("position",new hn(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const _=new xt(g,d),m=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Bh;let p=this.type;this.render=function(w,T,L){if(m.enabled===!1||m.autoUpdate===!1&&m.needsUpdate===!1||w.length===0)return;const y=n.getRenderTarget(),b=n.getActiveCubeFace(),N=n.getActiveMipmapLevel(),O=n.state;O.setBlending(si),O.buffers.depth.getReversed()===!0?O.buffers.color.setClear(0,0,0,0):O.buffers.color.setClear(1,1,1,1),O.buffers.depth.setTest(!0),O.setScissorTest(!1);const G=p!==Bn&&this.type===Bn,V=p===Bn&&this.type!==Bn;for(let X=0,q=w.length;X<q;X++){const J=w[X],z=J.shadow;if(z===void 0){console.warn("THREE.WebGLShadowMap:",J,"has no shadow.");continue}if(z.autoUpdate===!1&&z.needsUpdate===!1)continue;s.copy(z.mapSize);const H=z.getFrameExtents();if(s.multiply(H),r.copy(z.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/H.x),s.x=r.x*H.x,z.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/H.y),s.y=r.y*H.y,z.mapSize.y=r.y)),z.map===null||G===!0||V===!0){const W=this.type!==Bn?{minFilter:xn,magFilter:xn}:{};z.map!==null&&z.map.dispose(),z.map=new Ai(s.x,s.y,W),z.map.texture.name=J.name+".shadowMap",z.camera.updateProjectionMatrix()}n.setRenderTarget(z.map),n.clear();const D=z.getViewportCount();for(let W=0;W<D;W++){const Q=z.getViewport(W);o.set(r.x*Q.x,r.y*Q.y,r.x*Q.z,r.y*Q.w),O.viewport(o),z.updateMatrices(J,W),i=z.getFrustum(),x(T,L,z.camera,J,this.type)}z.isPointLightShadow!==!0&&this.type===Bn&&C(z,L),z.needsUpdate=!1}p=this.type,m.needsUpdate=!1,n.setRenderTarget(y,b,N)};function C(w,T){const L=e.update(_);d.defines.VSM_SAMPLES!==w.blurSamples&&(d.defines.VSM_SAMPLES=w.blurSamples,f.defines.VSM_SAMPLES=w.blurSamples,d.needsUpdate=!0,f.needsUpdate=!0),w.mapPass===null&&(w.mapPass=new Ai(s.x,s.y)),d.uniforms.shadow_pass.value=w.map.texture,d.uniforms.resolution.value=w.mapSize,d.uniforms.radius.value=w.radius,n.setRenderTarget(w.mapPass),n.clear(),n.renderBufferDirect(T,null,L,d,_,null),f.uniforms.shadow_pass.value=w.mapPass.texture,f.uniforms.resolution.value=w.mapSize,f.uniforms.radius.value=w.radius,n.setRenderTarget(w.map),n.clear(),n.renderBufferDirect(T,null,L,f,_,null)}function M(w,T,L,y){let b=null;const N=L.isPointLight===!0?w.customDistanceMaterial:w.customDepthMaterial;if(N!==void 0)b=N;else if(b=L.isPointLight===!0?l:a,n.localClippingEnabled&&T.clipShadows===!0&&Array.isArray(T.clippingPlanes)&&T.clippingPlanes.length!==0||T.displacementMap&&T.displacementScale!==0||T.alphaMap&&T.alphaTest>0||T.map&&T.alphaTest>0||T.alphaToCoverage===!0){const O=b.uuid,G=T.uuid;let V=c[O];V===void 0&&(V={},c[O]=V);let X=V[G];X===void 0&&(X=b.clone(),V[G]=X,T.addEventListener("dispose",P)),b=X}if(b.visible=T.visible,b.wireframe=T.wireframe,y===Bn?b.side=T.shadowSide!==null?T.shadowSide:T.side:b.side=T.shadowSide!==null?T.shadowSide:h[T.side],b.alphaMap=T.alphaMap,b.alphaTest=T.alphaToCoverage===!0?.5:T.alphaTest,b.map=T.map,b.clipShadows=T.clipShadows,b.clippingPlanes=T.clippingPlanes,b.clipIntersection=T.clipIntersection,b.displacementMap=T.displacementMap,b.displacementScale=T.displacementScale,b.displacementBias=T.displacementBias,b.wireframeLinewidth=T.wireframeLinewidth,b.linewidth=T.linewidth,L.isPointLight===!0&&b.isMeshDistanceMaterial===!0){const O=n.properties.get(b);O.light=L}return b}function x(w,T,L,y,b){if(w.visible===!1)return;if(w.layers.test(T.layers)&&(w.isMesh||w.isLine||w.isPoints)&&(w.castShadow||w.receiveShadow&&b===Bn)&&(!w.frustumCulled||i.intersectsObject(w))){w.modelViewMatrix.multiplyMatrices(L.matrixWorldInverse,w.matrixWorld);const G=e.update(w),V=w.material;if(Array.isArray(V)){const X=G.groups;for(let q=0,J=X.length;q<J;q++){const z=X[q],H=V[z.materialIndex];if(H&&H.visible){const D=M(w,H,y,b);w.onBeforeShadow(n,w,T,L,G,D,z),n.renderBufferDirect(L,null,G,D,w,z),w.onAfterShadow(n,w,T,L,G,D,z)}}}else if(V.visible){const X=M(w,V,y,b);w.onBeforeShadow(n,w,T,L,G,X,null),n.renderBufferDirect(L,null,G,X,w,null),w.onAfterShadow(n,w,T,L,G,X,null)}}const O=w.children;for(let G=0,V=O.length;G<V;G++)x(O[G],T,L,y,b)}function P(w){w.target.removeEventListener("dispose",P);for(const L in c){const y=c[L],b=w.target.uuid;b in y&&(y[b].dispose(),delete y[b])}}}const YS={[Ia]:Na,[Ua]:ka,[Oa]:za,[as]:Fa,[Na]:Ia,[ka]:Ua,[za]:Oa,[Fa]:as};function qS(n,e){function t(){let U=!1;const oe=new ct;let he=null;const ye=new ct(0,0,0,0);return{setMask:function(ie){he!==ie&&!U&&(n.colorMask(ie,ie,ie,ie),he=ie)},setLocked:function(ie){U=ie},setClear:function(ie,ee,be,Be,dt){dt===!0&&(ie*=Be,ee*=Be,be*=Be),oe.set(ie,ee,be,Be),ye.equals(oe)===!1&&(n.clearColor(ie,ee,be,Be),ye.copy(oe))},reset:function(){U=!1,he=null,ye.set(-1,0,0,0)}}}function i(){let U=!1,oe=!1,he=null,ye=null,ie=null;return{setReversed:function(ee){if(oe!==ee){const be=e.get("EXT_clip_control");ee?be.clipControlEXT(be.LOWER_LEFT_EXT,be.ZERO_TO_ONE_EXT):be.clipControlEXT(be.LOWER_LEFT_EXT,be.NEGATIVE_ONE_TO_ONE_EXT),oe=ee;const Be=ie;ie=null,this.setClear(Be)}},getReversed:function(){return oe},setTest:function(ee){ee?te(n.DEPTH_TEST):xe(n.DEPTH_TEST)},setMask:function(ee){he!==ee&&!U&&(n.depthMask(ee),he=ee)},setFunc:function(ee){if(oe&&(ee=YS[ee]),ye!==ee){switch(ee){case Ia:n.depthFunc(n.NEVER);break;case Na:n.depthFunc(n.ALWAYS);break;case Ua:n.depthFunc(n.LESS);break;case as:n.depthFunc(n.LEQUAL);break;case Oa:n.depthFunc(n.EQUAL);break;case Fa:n.depthFunc(n.GEQUAL);break;case ka:n.depthFunc(n.GREATER);break;case za:n.depthFunc(n.NOTEQUAL);break;default:n.depthFunc(n.LEQUAL)}ye=ee}},setLocked:function(ee){U=ee},setClear:function(ee){ie!==ee&&(oe&&(ee=1-ee),n.clearDepth(ee),ie=ee)},reset:function(){U=!1,he=null,ye=null,ie=null,oe=!1}}}function s(){let U=!1,oe=null,he=null,ye=null,ie=null,ee=null,be=null,Be=null,dt=null;return{setTest:function(nt){U||(nt?te(n.STENCIL_TEST):xe(n.STENCIL_TEST))},setMask:function(nt){oe!==nt&&!U&&(n.stencilMask(nt),oe=nt)},setFunc:function(nt,Nn,yn){(he!==nt||ye!==Nn||ie!==yn)&&(n.stencilFunc(nt,Nn,yn),he=nt,ye=Nn,ie=yn)},setOp:function(nt,Nn,yn){(ee!==nt||be!==Nn||Be!==yn)&&(n.stencilOp(nt,Nn,yn),ee=nt,be=Nn,Be=yn)},setLocked:function(nt){U=nt},setClear:function(nt){dt!==nt&&(n.clearStencil(nt),dt=nt)},reset:function(){U=!1,oe=null,he=null,ye=null,ie=null,ee=null,be=null,Be=null,dt=null}}}const r=new t,o=new i,a=new s,l=new WeakMap,c=new WeakMap;let u={},h={},d=new WeakMap,f=[],g=null,_=!1,m=null,p=null,C=null,M=null,x=null,P=null,w=null,T=new Qe(0,0,0),L=0,y=!1,b=null,N=null,O=null,G=null,V=null;const X=n.getParameter(n.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let q=!1,J=0;const z=n.getParameter(n.VERSION);z.indexOf("WebGL")!==-1?(J=parseFloat(/^WebGL (\d)/.exec(z)[1]),q=J>=1):z.indexOf("OpenGL ES")!==-1&&(J=parseFloat(/^OpenGL ES (\d)/.exec(z)[1]),q=J>=2);let H=null,D={};const W=n.getParameter(n.SCISSOR_BOX),Q=n.getParameter(n.VIEWPORT),me=new ct().fromArray(W),De=new ct().fromArray(Q);function Ie(U,oe,he,ye){const ie=new Uint8Array(4),ee=n.createTexture();n.bindTexture(U,ee),n.texParameteri(U,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(U,n.TEXTURE_MAG_FILTER,n.NEAREST);for(let be=0;be<he;be++)U===n.TEXTURE_3D||U===n.TEXTURE_2D_ARRAY?n.texImage3D(oe,0,n.RGBA,1,1,ye,0,n.RGBA,n.UNSIGNED_BYTE,ie):n.texImage2D(oe+be,0,n.RGBA,1,1,0,n.RGBA,n.UNSIGNED_BYTE,ie);return ee}const j={};j[n.TEXTURE_2D]=Ie(n.TEXTURE_2D,n.TEXTURE_2D,1),j[n.TEXTURE_CUBE_MAP]=Ie(n.TEXTURE_CUBE_MAP,n.TEXTURE_CUBE_MAP_POSITIVE_X,6),j[n.TEXTURE_2D_ARRAY]=Ie(n.TEXTURE_2D_ARRAY,n.TEXTURE_2D_ARRAY,1,1),j[n.TEXTURE_3D]=Ie(n.TEXTURE_3D,n.TEXTURE_3D,1,1),r.setClear(0,0,0,1),o.setClear(1),a.setClear(0),te(n.DEPTH_TEST),o.setFunc(as),ke(!1),Me(Hl),te(n.CULL_FACE),_t(si);function te(U){u[U]!==!0&&(n.enable(U),u[U]=!0)}function xe(U){u[U]!==!1&&(n.disable(U),u[U]=!1)}function Fe(U,oe){return h[U]!==oe?(n.bindFramebuffer(U,oe),h[U]=oe,U===n.DRAW_FRAMEBUFFER&&(h[n.FRAMEBUFFER]=oe),U===n.FRAMEBUFFER&&(h[n.DRAW_FRAMEBUFFER]=oe),!0):!1}function Te(U,oe){let he=f,ye=!1;if(U){he=d.get(oe),he===void 0&&(he=[],d.set(oe,he));const ie=U.textures;if(he.length!==ie.length||he[0]!==n.COLOR_ATTACHMENT0){for(let ee=0,be=ie.length;ee<be;ee++)he[ee]=n.COLOR_ATTACHMENT0+ee;he.length=ie.length,ye=!0}}else he[0]!==n.BACK&&(he[0]=n.BACK,ye=!0);ye&&n.drawBuffers(he)}function Je(U){return g!==U?(n.useProgram(U),g=U,!0):!1}const kt={[yi]:n.FUNC_ADD,[fg]:n.FUNC_SUBTRACT,[pg]:n.FUNC_REVERSE_SUBTRACT};kt[mg]=n.MIN,kt[gg]=n.MAX;const I={[_g]:n.ZERO,[vg]:n.ONE,[xg]:n.SRC_COLOR,[La]:n.SRC_ALPHA,[Eg]:n.SRC_ALPHA_SATURATE,[Mg]:n.DST_COLOR,[Sg]:n.DST_ALPHA,[yg]:n.ONE_MINUS_SRC_COLOR,[Da]:n.ONE_MINUS_SRC_ALPHA,[wg]:n.ONE_MINUS_DST_COLOR,[bg]:n.ONE_MINUS_DST_ALPHA,[Tg]:n.CONSTANT_COLOR,[Ag]:n.ONE_MINUS_CONSTANT_COLOR,[Cg]:n.CONSTANT_ALPHA,[Rg]:n.ONE_MINUS_CONSTANT_ALPHA};function _t(U,oe,he,ye,ie,ee,be,Be,dt,nt){if(U===si){_===!0&&(xe(n.BLEND),_=!1);return}if(_===!1&&(te(n.BLEND),_=!0),U!==dg){if(U!==m||nt!==y){if((p!==yi||x!==yi)&&(n.blendEquation(n.FUNC_ADD),p=yi,x=yi),nt)switch(U){case ns:n.blendFuncSeparate(n.ONE,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case Gl:n.blendFunc(n.ONE,n.ONE);break;case Vl:n.blendFuncSeparate(n.ZERO,n.ONE_MINUS_SRC_COLOR,n.ZERO,n.ONE);break;case Wl:n.blendFuncSeparate(n.DST_COLOR,n.ONE_MINUS_SRC_ALPHA,n.ZERO,n.ONE);break;default:console.error("THREE.WebGLState: Invalid blending: ",U);break}else switch(U){case ns:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE_MINUS_SRC_ALPHA,n.ONE,n.ONE_MINUS_SRC_ALPHA);break;case Gl:n.blendFuncSeparate(n.SRC_ALPHA,n.ONE,n.ONE,n.ONE);break;case Vl:console.error("THREE.WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case Wl:console.error("THREE.WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:console.error("THREE.WebGLState: Invalid blending: ",U);break}C=null,M=null,P=null,w=null,T.set(0,0,0),L=0,m=U,y=nt}return}ie=ie||oe,ee=ee||he,be=be||ye,(oe!==p||ie!==x)&&(n.blendEquationSeparate(kt[oe],kt[ie]),p=oe,x=ie),(he!==C||ye!==M||ee!==P||be!==w)&&(n.blendFuncSeparate(I[he],I[ye],I[ee],I[be]),C=he,M=ye,P=ee,w=be),(Be.equals(T)===!1||dt!==L)&&(n.blendColor(Be.r,Be.g,Be.b,dt),T.copy(Be),L=dt),m=U,y=!1}function He(U,oe){U.side===cn?xe(n.CULL_FACE):te(n.CULL_FACE);let he=U.side===Jt;oe&&(he=!he),ke(he),U.blending===ns&&U.transparent===!1?_t(si):_t(U.blending,U.blendEquation,U.blendSrc,U.blendDst,U.blendEquationAlpha,U.blendSrcAlpha,U.blendDstAlpha,U.blendColor,U.blendAlpha,U.premultipliedAlpha),o.setFunc(U.depthFunc),o.setTest(U.depthTest),o.setMask(U.depthWrite),r.setMask(U.colorWrite);const ye=U.stencilWrite;a.setTest(ye),ye&&(a.setMask(U.stencilWriteMask),a.setFunc(U.stencilFunc,U.stencilRef,U.stencilFuncMask),a.setOp(U.stencilFail,U.stencilZFail,U.stencilZPass)),we(U.polygonOffset,U.polygonOffsetFactor,U.polygonOffsetUnits),U.alphaToCoverage===!0?te(n.SAMPLE_ALPHA_TO_COVERAGE):xe(n.SAMPLE_ALPHA_TO_COVERAGE)}function ke(U){b!==U&&(U?n.frontFace(n.CW):n.frontFace(n.CCW),b=U)}function Me(U){U!==lg?(te(n.CULL_FACE),U!==N&&(U===Hl?n.cullFace(n.BACK):U===ug?n.cullFace(n.FRONT):n.cullFace(n.FRONT_AND_BACK))):xe(n.CULL_FACE),N=U}function vt(U){U!==O&&(q&&n.lineWidth(U),O=U)}function we(U,oe,he){U?(te(n.POLYGON_OFFSET_FILL),(G!==oe||V!==he)&&(n.polygonOffset(oe,he),G=oe,V=he)):xe(n.POLYGON_OFFSET_FILL)}function Xe(U){U?te(n.SCISSOR_TEST):xe(n.SCISSOR_TEST)}function Nt(U){U===void 0&&(U=n.TEXTURE0+X-1),H!==U&&(n.activeTexture(U),H=U)}function wt(U,oe,he){he===void 0&&(H===null?he=n.TEXTURE0+X-1:he=H);let ye=D[he];ye===void 0&&(ye={type:void 0,texture:void 0},D[he]=ye),(ye.type!==U||ye.texture!==oe)&&(H!==he&&(n.activeTexture(he),H=he),n.bindTexture(U,oe||j[U]),ye.type=U,ye.texture=oe)}function A(){const U=D[H];U!==void 0&&U.type!==void 0&&(n.bindTexture(U.type,null),U.type=void 0,U.texture=void 0)}function v(){try{n.compressedTexImage2D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function B(){try{n.compressedTexImage3D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function K(){try{n.texSubImage2D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function ne(){try{n.texSubImage3D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Z(){try{n.compressedTexSubImage2D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Re(){try{n.compressedTexSubImage3D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function ce(){try{n.texStorage2D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Ee(){try{n.texStorage3D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function Ae(){try{n.texImage2D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function re(){try{n.texImage3D(...arguments)}catch(U){console.error("THREE.WebGLState:",U)}}function ge(U){me.equals(U)===!1&&(n.scissor(U.x,U.y,U.z,U.w),me.copy(U))}function Oe(U){De.equals(U)===!1&&(n.viewport(U.x,U.y,U.z,U.w),De.copy(U))}function Ce(U,oe){let he=c.get(oe);he===void 0&&(he=new WeakMap,c.set(oe,he));let ye=he.get(U);ye===void 0&&(ye=n.getUniformBlockIndex(oe,U.name),he.set(U,ye))}function fe(U,oe){const ye=c.get(oe).get(U);l.get(oe)!==ye&&(n.uniformBlockBinding(oe,ye,U.__bindingPointIndex),l.set(oe,ye))}function Ge(){n.disable(n.BLEND),n.disable(n.CULL_FACE),n.disable(n.DEPTH_TEST),n.disable(n.POLYGON_OFFSET_FILL),n.disable(n.SCISSOR_TEST),n.disable(n.STENCIL_TEST),n.disable(n.SAMPLE_ALPHA_TO_COVERAGE),n.blendEquation(n.FUNC_ADD),n.blendFunc(n.ONE,n.ZERO),n.blendFuncSeparate(n.ONE,n.ZERO,n.ONE,n.ZERO),n.blendColor(0,0,0,0),n.colorMask(!0,!0,!0,!0),n.clearColor(0,0,0,0),n.depthMask(!0),n.depthFunc(n.LESS),o.setReversed(!1),n.clearDepth(1),n.stencilMask(4294967295),n.stencilFunc(n.ALWAYS,0,4294967295),n.stencilOp(n.KEEP,n.KEEP,n.KEEP),n.clearStencil(0),n.cullFace(n.BACK),n.frontFace(n.CCW),n.polygonOffset(0,0),n.activeTexture(n.TEXTURE0),n.bindFramebuffer(n.FRAMEBUFFER,null),n.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),n.bindFramebuffer(n.READ_FRAMEBUFFER,null),n.useProgram(null),n.lineWidth(1),n.scissor(0,0,n.canvas.width,n.canvas.height),n.viewport(0,0,n.canvas.width,n.canvas.height),u={},H=null,D={},h={},d=new WeakMap,f=[],g=null,_=!1,m=null,p=null,C=null,M=null,x=null,P=null,w=null,T=new Qe(0,0,0),L=0,y=!1,b=null,N=null,O=null,G=null,V=null,me.set(0,0,n.canvas.width,n.canvas.height),De.set(0,0,n.canvas.width,n.canvas.height),r.reset(),o.reset(),a.reset()}return{buffers:{color:r,depth:o,stencil:a},enable:te,disable:xe,bindFramebuffer:Fe,drawBuffers:Te,useProgram:Je,setBlending:_t,setMaterial:He,setFlipSided:ke,setCullFace:Me,setLineWidth:vt,setPolygonOffset:we,setScissorTest:Xe,activeTexture:Nt,bindTexture:wt,unbindTexture:A,compressedTexImage2D:v,compressedTexImage3D:B,texImage2D:Ae,texImage3D:re,updateUBOMapping:Ce,uniformBlockBinding:fe,texStorage2D:ce,texStorage3D:Ee,texSubImage2D:K,texSubImage3D:ne,compressedTexSubImage2D:Z,compressedTexSubImage3D:Re,scissor:ge,viewport:Oe,reset:Ge}}function jS(n,e,t,i,s,r,o){const a=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,l=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),c=new ue,u=new WeakMap;let h;const d=new WeakMap;let f=!1;try{f=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function g(A,v){return f?new OffscreenCanvas(A,v):Ws("canvas")}function _(A,v,B){let K=1;const ne=wt(A);if((ne.width>B||ne.height>B)&&(K=B/Math.max(ne.width,ne.height)),K<1)if(typeof HTMLImageElement<"u"&&A instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&A instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&A instanceof ImageBitmap||typeof VideoFrame<"u"&&A instanceof VideoFrame){const Z=Math.floor(K*ne.width),Re=Math.floor(K*ne.height);h===void 0&&(h=g(Z,Re));const ce=v?g(Z,Re):h;return ce.width=Z,ce.height=Re,ce.getContext("2d").drawImage(A,0,0,Z,Re),console.warn("THREE.WebGLRenderer: Texture has been resized from ("+ne.width+"x"+ne.height+") to ("+Z+"x"+Re+")."),ce}else return"data"in A&&console.warn("THREE.WebGLRenderer: Image in DataTexture is too big ("+ne.width+"x"+ne.height+")."),A;return A}function m(A){return A.generateMipmaps}function p(A){n.generateMipmap(A)}function C(A){return A.isWebGLCubeRenderTarget?n.TEXTURE_CUBE_MAP:A.isWebGL3DRenderTarget?n.TEXTURE_3D:A.isWebGLArrayRenderTarget||A.isCompressedArrayTexture?n.TEXTURE_2D_ARRAY:n.TEXTURE_2D}function M(A,v,B,K,ne=!1){if(A!==null){if(n[A]!==void 0)return n[A];console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '"+A+"'")}let Z=v;if(v===n.RED&&(B===n.FLOAT&&(Z=n.R32F),B===n.HALF_FLOAT&&(Z=n.R16F),B===n.UNSIGNED_BYTE&&(Z=n.R8)),v===n.RED_INTEGER&&(B===n.UNSIGNED_BYTE&&(Z=n.R8UI),B===n.UNSIGNED_SHORT&&(Z=n.R16UI),B===n.UNSIGNED_INT&&(Z=n.R32UI),B===n.BYTE&&(Z=n.R8I),B===n.SHORT&&(Z=n.R16I),B===n.INT&&(Z=n.R32I)),v===n.RG&&(B===n.FLOAT&&(Z=n.RG32F),B===n.HALF_FLOAT&&(Z=n.RG16F),B===n.UNSIGNED_BYTE&&(Z=n.RG8)),v===n.RG_INTEGER&&(B===n.UNSIGNED_BYTE&&(Z=n.RG8UI),B===n.UNSIGNED_SHORT&&(Z=n.RG16UI),B===n.UNSIGNED_INT&&(Z=n.RG32UI),B===n.BYTE&&(Z=n.RG8I),B===n.SHORT&&(Z=n.RG16I),B===n.INT&&(Z=n.RG32I)),v===n.RGB_INTEGER&&(B===n.UNSIGNED_BYTE&&(Z=n.RGB8UI),B===n.UNSIGNED_SHORT&&(Z=n.RGB16UI),B===n.UNSIGNED_INT&&(Z=n.RGB32UI),B===n.BYTE&&(Z=n.RGB8I),B===n.SHORT&&(Z=n.RGB16I),B===n.INT&&(Z=n.RGB32I)),v===n.RGBA_INTEGER&&(B===n.UNSIGNED_BYTE&&(Z=n.RGBA8UI),B===n.UNSIGNED_SHORT&&(Z=n.RGBA16UI),B===n.UNSIGNED_INT&&(Z=n.RGBA32UI),B===n.BYTE&&(Z=n.RGBA8I),B===n.SHORT&&(Z=n.RGBA16I),B===n.INT&&(Z=n.RGBA32I)),v===n.RGB&&(B===n.UNSIGNED_INT_5_9_9_9_REV&&(Z=n.RGB9_E5),B===n.UNSIGNED_INT_10F_11F_11F_REV&&(Z=n.R11F_G11F_B10F)),v===n.RGBA){const Re=ne?oo:tt.getTransfer(K);B===n.FLOAT&&(Z=n.RGBA32F),B===n.HALF_FLOAT&&(Z=n.RGBA16F),B===n.UNSIGNED_BYTE&&(Z=Re===rt?n.SRGB8_ALPHA8:n.RGBA8),B===n.UNSIGNED_SHORT_4_4_4_4&&(Z=n.RGBA4),B===n.UNSIGNED_SHORT_5_5_5_1&&(Z=n.RGB5_A1)}return(Z===n.R16F||Z===n.R32F||Z===n.RG16F||Z===n.RG32F||Z===n.RGBA16F||Z===n.RGBA32F)&&e.get("EXT_color_buffer_float"),Z}function x(A,v){let B;return A?v===null||v===Ei||v===Hs?B=n.DEPTH24_STENCIL8:v===Wn?B=n.DEPTH32F_STENCIL8:v===Bs&&(B=n.DEPTH24_STENCIL8,console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):v===null||v===Ei||v===Hs?B=n.DEPTH_COMPONENT24:v===Wn?B=n.DEPTH_COMPONENT32F:v===Bs&&(B=n.DEPTH_COMPONENT16),B}function P(A,v){return m(A)===!0||A.isFramebufferTexture&&A.minFilter!==xn&&A.minFilter!==Tn?Math.log2(Math.max(v.width,v.height))+1:A.mipmaps!==void 0&&A.mipmaps.length>0?A.mipmaps.length:A.isCompressedTexture&&Array.isArray(A.image)?v.mipmaps.length:1}function w(A){const v=A.target;v.removeEventListener("dispose",w),L(v),v.isVideoTexture&&u.delete(v)}function T(A){const v=A.target;v.removeEventListener("dispose",T),b(v)}function L(A){const v=i.get(A);if(v.__webglInit===void 0)return;const B=A.source,K=d.get(B);if(K){const ne=K[v.__cacheKey];ne.usedTimes--,ne.usedTimes===0&&y(A),Object.keys(K).length===0&&d.delete(B)}i.remove(A)}function y(A){const v=i.get(A);n.deleteTexture(v.__webglTexture);const B=A.source,K=d.get(B);delete K[v.__cacheKey],o.memory.textures--}function b(A){const v=i.get(A);if(A.depthTexture&&(A.depthTexture.dispose(),i.remove(A.depthTexture)),A.isWebGLCubeRenderTarget)for(let K=0;K<6;K++){if(Array.isArray(v.__webglFramebuffer[K]))for(let ne=0;ne<v.__webglFramebuffer[K].length;ne++)n.deleteFramebuffer(v.__webglFramebuffer[K][ne]);else n.deleteFramebuffer(v.__webglFramebuffer[K]);v.__webglDepthbuffer&&n.deleteRenderbuffer(v.__webglDepthbuffer[K])}else{if(Array.isArray(v.__webglFramebuffer))for(let K=0;K<v.__webglFramebuffer.length;K++)n.deleteFramebuffer(v.__webglFramebuffer[K]);else n.deleteFramebuffer(v.__webglFramebuffer);if(v.__webglDepthbuffer&&n.deleteRenderbuffer(v.__webglDepthbuffer),v.__webglMultisampledFramebuffer&&n.deleteFramebuffer(v.__webglMultisampledFramebuffer),v.__webglColorRenderbuffer)for(let K=0;K<v.__webglColorRenderbuffer.length;K++)v.__webglColorRenderbuffer[K]&&n.deleteRenderbuffer(v.__webglColorRenderbuffer[K]);v.__webglDepthRenderbuffer&&n.deleteRenderbuffer(v.__webglDepthRenderbuffer)}const B=A.textures;for(let K=0,ne=B.length;K<ne;K++){const Z=i.get(B[K]);Z.__webglTexture&&(n.deleteTexture(Z.__webglTexture),o.memory.textures--),i.remove(B[K])}i.remove(A)}let N=0;function O(){N=0}function G(){const A=N;return A>=s.maxTextures&&console.warn("THREE.WebGLTextures: Trying to use "+A+" texture units while this GPU supports only "+s.maxTextures),N+=1,A}function V(A){const v=[];return v.push(A.wrapS),v.push(A.wrapT),v.push(A.wrapR||0),v.push(A.magFilter),v.push(A.minFilter),v.push(A.anisotropy),v.push(A.internalFormat),v.push(A.format),v.push(A.type),v.push(A.generateMipmaps),v.push(A.premultiplyAlpha),v.push(A.flipY),v.push(A.unpackAlignment),v.push(A.colorSpace),v.join()}function X(A,v){const B=i.get(A);if(A.isVideoTexture&&Xe(A),A.isRenderTargetTexture===!1&&A.isExternalTexture!==!0&&A.version>0&&B.__version!==A.version){const K=A.image;if(K===null)console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");else if(K.complete===!1)console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");else{j(B,A,v);return}}else A.isExternalTexture&&(B.__webglTexture=A.sourceTexture?A.sourceTexture:null);t.bindTexture(n.TEXTURE_2D,B.__webglTexture,n.TEXTURE0+v)}function q(A,v){const B=i.get(A);if(A.isRenderTargetTexture===!1&&A.version>0&&B.__version!==A.version){j(B,A,v);return}t.bindTexture(n.TEXTURE_2D_ARRAY,B.__webglTexture,n.TEXTURE0+v)}function J(A,v){const B=i.get(A);if(A.isRenderTargetTexture===!1&&A.version>0&&B.__version!==A.version){j(B,A,v);return}t.bindTexture(n.TEXTURE_3D,B.__webglTexture,n.TEXTURE0+v)}function z(A,v){const B=i.get(A);if(A.version>0&&B.__version!==A.version){te(B,A,v);return}t.bindTexture(n.TEXTURE_CUBE_MAP,B.__webglTexture,n.TEXTURE0+v)}const H={[Ga]:n.REPEAT,[Mi]:n.CLAMP_TO_EDGE,[Va]:n.MIRRORED_REPEAT},D={[xn]:n.NEAREST,[zg]:n.NEAREST_MIPMAP_NEAREST,[dr]:n.NEAREST_MIPMAP_LINEAR,[Tn]:n.LINEAR,[Io]:n.LINEAR_MIPMAP_NEAREST,[wi]:n.LINEAR_MIPMAP_LINEAR},W={[Wg]:n.NEVER,[Zg]:n.ALWAYS,[Xg]:n.LESS,[Jh]:n.LEQUAL,[$g]:n.EQUAL,[jg]:n.GEQUAL,[Yg]:n.GREATER,[qg]:n.NOTEQUAL};function Q(A,v){if(v.type===Wn&&e.has("OES_texture_float_linear")===!1&&(v.magFilter===Tn||v.magFilter===Io||v.magFilter===dr||v.magFilter===wi||v.minFilter===Tn||v.minFilter===Io||v.minFilter===dr||v.minFilter===wi)&&console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),n.texParameteri(A,n.TEXTURE_WRAP_S,H[v.wrapS]),n.texParameteri(A,n.TEXTURE_WRAP_T,H[v.wrapT]),(A===n.TEXTURE_3D||A===n.TEXTURE_2D_ARRAY)&&n.texParameteri(A,n.TEXTURE_WRAP_R,H[v.wrapR]),n.texParameteri(A,n.TEXTURE_MAG_FILTER,D[v.magFilter]),n.texParameteri(A,n.TEXTURE_MIN_FILTER,D[v.minFilter]),v.compareFunction&&(n.texParameteri(A,n.TEXTURE_COMPARE_MODE,n.COMPARE_REF_TO_TEXTURE),n.texParameteri(A,n.TEXTURE_COMPARE_FUNC,W[v.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(v.magFilter===xn||v.minFilter!==dr&&v.minFilter!==wi||v.type===Wn&&e.has("OES_texture_float_linear")===!1)return;if(v.anisotropy>1||i.get(v).__currentAnisotropy){const B=e.get("EXT_texture_filter_anisotropic");n.texParameterf(A,B.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,s.getMaxAnisotropy())),i.get(v).__currentAnisotropy=v.anisotropy}}}function me(A,v){let B=!1;A.__webglInit===void 0&&(A.__webglInit=!0,v.addEventListener("dispose",w));const K=v.source;let ne=d.get(K);ne===void 0&&(ne={},d.set(K,ne));const Z=V(v);if(Z!==A.__cacheKey){ne[Z]===void 0&&(ne[Z]={texture:n.createTexture(),usedTimes:0},o.memory.textures++,B=!0),ne[Z].usedTimes++;const Re=ne[A.__cacheKey];Re!==void 0&&(ne[A.__cacheKey].usedTimes--,Re.usedTimes===0&&y(v)),A.__cacheKey=Z,A.__webglTexture=ne[Z].texture}return B}function De(A,v,B){return Math.floor(Math.floor(A/B)/v)}function Ie(A,v,B,K){const Z=A.updateRanges;if(Z.length===0)t.texSubImage2D(n.TEXTURE_2D,0,0,0,v.width,v.height,B,K,v.data);else{Z.sort((re,ge)=>re.start-ge.start);let Re=0;for(let re=1;re<Z.length;re++){const ge=Z[Re],Oe=Z[re],Ce=ge.start+ge.count,fe=De(Oe.start,v.width,4),Ge=De(ge.start,v.width,4);Oe.start<=Ce+1&&fe===Ge&&De(Oe.start+Oe.count-1,v.width,4)===fe?ge.count=Math.max(ge.count,Oe.start+Oe.count-ge.start):(++Re,Z[Re]=Oe)}Z.length=Re+1;const ce=n.getParameter(n.UNPACK_ROW_LENGTH),Ee=n.getParameter(n.UNPACK_SKIP_PIXELS),Ae=n.getParameter(n.UNPACK_SKIP_ROWS);n.pixelStorei(n.UNPACK_ROW_LENGTH,v.width);for(let re=0,ge=Z.length;re<ge;re++){const Oe=Z[re],Ce=Math.floor(Oe.start/4),fe=Math.ceil(Oe.count/4),Ge=Ce%v.width,U=Math.floor(Ce/v.width),oe=fe,he=1;n.pixelStorei(n.UNPACK_SKIP_PIXELS,Ge),n.pixelStorei(n.UNPACK_SKIP_ROWS,U),t.texSubImage2D(n.TEXTURE_2D,0,Ge,U,oe,he,B,K,v.data)}A.clearUpdateRanges(),n.pixelStorei(n.UNPACK_ROW_LENGTH,ce),n.pixelStorei(n.UNPACK_SKIP_PIXELS,Ee),n.pixelStorei(n.UNPACK_SKIP_ROWS,Ae)}}function j(A,v,B){let K=n.TEXTURE_2D;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&(K=n.TEXTURE_2D_ARRAY),v.isData3DTexture&&(K=n.TEXTURE_3D);const ne=me(A,v),Z=v.source;t.bindTexture(K,A.__webglTexture,n.TEXTURE0+B);const Re=i.get(Z);if(Z.version!==Re.__version||ne===!0){t.activeTexture(n.TEXTURE0+B);const ce=tt.getPrimaries(tt.workingColorSpace),Ee=v.colorSpace===ti?null:tt.getPrimaries(v.colorSpace),Ae=v.colorSpace===ti||ce===Ee?n.NONE:n.BROWSER_DEFAULT_WEBGL;n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,v.flipY),n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),n.pixelStorei(n.UNPACK_ALIGNMENT,v.unpackAlignment),n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ae);let re=_(v.image,!1,s.maxTextureSize);re=Nt(v,re);const ge=r.convert(v.format,v.colorSpace),Oe=r.convert(v.type);let Ce=M(v.internalFormat,ge,Oe,v.colorSpace,v.isVideoTexture);Q(K,v);let fe;const Ge=v.mipmaps,U=v.isVideoTexture!==!0,oe=Re.__version===void 0||ne===!0,he=Z.dataReady,ye=P(v,re);if(v.isDepthTexture)Ce=x(v.format===Vs,v.type),oe&&(U?t.texStorage2D(n.TEXTURE_2D,1,Ce,re.width,re.height):t.texImage2D(n.TEXTURE_2D,0,Ce,re.width,re.height,0,ge,Oe,null));else if(v.isDataTexture)if(Ge.length>0){U&&oe&&t.texStorage2D(n.TEXTURE_2D,ye,Ce,Ge[0].width,Ge[0].height);for(let ie=0,ee=Ge.length;ie<ee;ie++)fe=Ge[ie],U?he&&t.texSubImage2D(n.TEXTURE_2D,ie,0,0,fe.width,fe.height,ge,Oe,fe.data):t.texImage2D(n.TEXTURE_2D,ie,Ce,fe.width,fe.height,0,ge,Oe,fe.data);v.generateMipmaps=!1}else U?(oe&&t.texStorage2D(n.TEXTURE_2D,ye,Ce,re.width,re.height),he&&Ie(v,re,ge,Oe)):t.texImage2D(n.TEXTURE_2D,0,Ce,re.width,re.height,0,ge,Oe,re.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){U&&oe&&t.texStorage3D(n.TEXTURE_2D_ARRAY,ye,Ce,Ge[0].width,Ge[0].height,re.depth);for(let ie=0,ee=Ge.length;ie<ee;ie++)if(fe=Ge[ie],v.format!==vn)if(ge!==null)if(U){if(he)if(v.layerUpdates.size>0){const be=Pu(fe.width,fe.height,v.format,v.type);for(const Be of v.layerUpdates){const dt=fe.data.subarray(Be*be/fe.data.BYTES_PER_ELEMENT,(Be+1)*be/fe.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,ie,0,0,Be,fe.width,fe.height,1,ge,dt)}v.clearLayerUpdates()}else t.compressedTexSubImage3D(n.TEXTURE_2D_ARRAY,ie,0,0,0,fe.width,fe.height,re.depth,ge,fe.data)}else t.compressedTexImage3D(n.TEXTURE_2D_ARRAY,ie,Ce,fe.width,fe.height,re.depth,0,fe.data,0,0);else console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else U?he&&t.texSubImage3D(n.TEXTURE_2D_ARRAY,ie,0,0,0,fe.width,fe.height,re.depth,ge,Oe,fe.data):t.texImage3D(n.TEXTURE_2D_ARRAY,ie,Ce,fe.width,fe.height,re.depth,0,ge,Oe,fe.data)}else{U&&oe&&t.texStorage2D(n.TEXTURE_2D,ye,Ce,Ge[0].width,Ge[0].height);for(let ie=0,ee=Ge.length;ie<ee;ie++)fe=Ge[ie],v.format!==vn?ge!==null?U?he&&t.compressedTexSubImage2D(n.TEXTURE_2D,ie,0,0,fe.width,fe.height,ge,fe.data):t.compressedTexImage2D(n.TEXTURE_2D,ie,Ce,fe.width,fe.height,0,fe.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):U?he&&t.texSubImage2D(n.TEXTURE_2D,ie,0,0,fe.width,fe.height,ge,Oe,fe.data):t.texImage2D(n.TEXTURE_2D,ie,Ce,fe.width,fe.height,0,ge,Oe,fe.data)}else if(v.isDataArrayTexture)if(U){if(oe&&t.texStorage3D(n.TEXTURE_2D_ARRAY,ye,Ce,re.width,re.height,re.depth),he)if(v.layerUpdates.size>0){const ie=Pu(re.width,re.height,v.format,v.type);for(const ee of v.layerUpdates){const be=re.data.subarray(ee*ie/re.data.BYTES_PER_ELEMENT,(ee+1)*ie/re.data.BYTES_PER_ELEMENT);t.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,ee,re.width,re.height,1,ge,Oe,be)}v.clearLayerUpdates()}else t.texSubImage3D(n.TEXTURE_2D_ARRAY,0,0,0,0,re.width,re.height,re.depth,ge,Oe,re.data)}else t.texImage3D(n.TEXTURE_2D_ARRAY,0,Ce,re.width,re.height,re.depth,0,ge,Oe,re.data);else if(v.isData3DTexture)U?(oe&&t.texStorage3D(n.TEXTURE_3D,ye,Ce,re.width,re.height,re.depth),he&&t.texSubImage3D(n.TEXTURE_3D,0,0,0,0,re.width,re.height,re.depth,ge,Oe,re.data)):t.texImage3D(n.TEXTURE_3D,0,Ce,re.width,re.height,re.depth,0,ge,Oe,re.data);else if(v.isFramebufferTexture){if(oe)if(U)t.texStorage2D(n.TEXTURE_2D,ye,Ce,re.width,re.height);else{let ie=re.width,ee=re.height;for(let be=0;be<ye;be++)t.texImage2D(n.TEXTURE_2D,be,Ce,ie,ee,0,ge,Oe,null),ie>>=1,ee>>=1}}else if(Ge.length>0){if(U&&oe){const ie=wt(Ge[0]);t.texStorage2D(n.TEXTURE_2D,ye,Ce,ie.width,ie.height)}for(let ie=0,ee=Ge.length;ie<ee;ie++)fe=Ge[ie],U?he&&t.texSubImage2D(n.TEXTURE_2D,ie,0,0,ge,Oe,fe):t.texImage2D(n.TEXTURE_2D,ie,Ce,ge,Oe,fe);v.generateMipmaps=!1}else if(U){if(oe){const ie=wt(re);t.texStorage2D(n.TEXTURE_2D,ye,Ce,ie.width,ie.height)}he&&t.texSubImage2D(n.TEXTURE_2D,0,0,0,ge,Oe,re)}else t.texImage2D(n.TEXTURE_2D,0,Ce,ge,Oe,re);m(v)&&p(K),Re.__version=Z.version,v.onUpdate&&v.onUpdate(v)}A.__version=v.version}function te(A,v,B){if(v.image.length!==6)return;const K=me(A,v),ne=v.source;t.bindTexture(n.TEXTURE_CUBE_MAP,A.__webglTexture,n.TEXTURE0+B);const Z=i.get(ne);if(ne.version!==Z.__version||K===!0){t.activeTexture(n.TEXTURE0+B);const Re=tt.getPrimaries(tt.workingColorSpace),ce=v.colorSpace===ti?null:tt.getPrimaries(v.colorSpace),Ee=v.colorSpace===ti||Re===ce?n.NONE:n.BROWSER_DEFAULT_WEBGL;n.pixelStorei(n.UNPACK_FLIP_Y_WEBGL,v.flipY),n.pixelStorei(n.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),n.pixelStorei(n.UNPACK_ALIGNMENT,v.unpackAlignment),n.pixelStorei(n.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ee);const Ae=v.isCompressedTexture||v.image[0].isCompressedTexture,re=v.image[0]&&v.image[0].isDataTexture,ge=[];for(let ee=0;ee<6;ee++)!Ae&&!re?ge[ee]=_(v.image[ee],!0,s.maxCubemapSize):ge[ee]=re?v.image[ee].image:v.image[ee],ge[ee]=Nt(v,ge[ee]);const Oe=ge[0],Ce=r.convert(v.format,v.colorSpace),fe=r.convert(v.type),Ge=M(v.internalFormat,Ce,fe,v.colorSpace),U=v.isVideoTexture!==!0,oe=Z.__version===void 0||K===!0,he=ne.dataReady;let ye=P(v,Oe);Q(n.TEXTURE_CUBE_MAP,v);let ie;if(Ae){U&&oe&&t.texStorage2D(n.TEXTURE_CUBE_MAP,ye,Ge,Oe.width,Oe.height);for(let ee=0;ee<6;ee++){ie=ge[ee].mipmaps;for(let be=0;be<ie.length;be++){const Be=ie[be];v.format!==vn?Ce!==null?U?he&&t.compressedTexSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be,0,0,Be.width,Be.height,Ce,Be.data):t.compressedTexImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be,Ge,Be.width,Be.height,0,Be.data):console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):U?he&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be,0,0,Be.width,Be.height,Ce,fe,Be.data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be,Ge,Be.width,Be.height,0,Ce,fe,Be.data)}}}else{if(ie=v.mipmaps,U&&oe){ie.length>0&&ye++;const ee=wt(ge[0]);t.texStorage2D(n.TEXTURE_CUBE_MAP,ye,Ge,ee.width,ee.height)}for(let ee=0;ee<6;ee++)if(re){U?he&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,0,0,ge[ee].width,ge[ee].height,Ce,fe,ge[ee].data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,Ge,ge[ee].width,ge[ee].height,0,Ce,fe,ge[ee].data);for(let be=0;be<ie.length;be++){const dt=ie[be].image[ee].image;U?he&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be+1,0,0,dt.width,dt.height,Ce,fe,dt.data):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be+1,Ge,dt.width,dt.height,0,Ce,fe,dt.data)}}else{U?he&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,0,0,Ce,fe,ge[ee]):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,0,Ge,Ce,fe,ge[ee]);for(let be=0;be<ie.length;be++){const Be=ie[be];U?he&&t.texSubImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be+1,0,0,Ce,fe,Be.image[ee]):t.texImage2D(n.TEXTURE_CUBE_MAP_POSITIVE_X+ee,be+1,Ge,Ce,fe,Be.image[ee])}}}m(v)&&p(n.TEXTURE_CUBE_MAP),Z.__version=ne.version,v.onUpdate&&v.onUpdate(v)}A.__version=v.version}function xe(A,v,B,K,ne,Z){const Re=r.convert(B.format,B.colorSpace),ce=r.convert(B.type),Ee=M(B.internalFormat,Re,ce,B.colorSpace),Ae=i.get(v),re=i.get(B);if(re.__renderTarget=v,!Ae.__hasExternalTextures){const ge=Math.max(1,v.width>>Z),Oe=Math.max(1,v.height>>Z);ne===n.TEXTURE_3D||ne===n.TEXTURE_2D_ARRAY?t.texImage3D(ne,Z,Ee,ge,Oe,v.depth,0,Re,ce,null):t.texImage2D(ne,Z,Ee,ge,Oe,0,Re,ce,null)}t.bindFramebuffer(n.FRAMEBUFFER,A),we(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,K,ne,re.__webglTexture,0,vt(v)):(ne===n.TEXTURE_2D||ne>=n.TEXTURE_CUBE_MAP_POSITIVE_X&&ne<=n.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&n.framebufferTexture2D(n.FRAMEBUFFER,K,ne,re.__webglTexture,Z),t.bindFramebuffer(n.FRAMEBUFFER,null)}function Fe(A,v,B){if(n.bindRenderbuffer(n.RENDERBUFFER,A),v.depthBuffer){const K=v.depthTexture,ne=K&&K.isDepthTexture?K.type:null,Z=x(v.stencilBuffer,ne),Re=v.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,ce=vt(v);we(v)?a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,ce,Z,v.width,v.height):B?n.renderbufferStorageMultisample(n.RENDERBUFFER,ce,Z,v.width,v.height):n.renderbufferStorage(n.RENDERBUFFER,Z,v.width,v.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,Re,n.RENDERBUFFER,A)}else{const K=v.textures;for(let ne=0;ne<K.length;ne++){const Z=K[ne],Re=r.convert(Z.format,Z.colorSpace),ce=r.convert(Z.type),Ee=M(Z.internalFormat,Re,ce,Z.colorSpace),Ae=vt(v);B&&we(v)===!1?n.renderbufferStorageMultisample(n.RENDERBUFFER,Ae,Ee,v.width,v.height):we(v)?a.renderbufferStorageMultisampleEXT(n.RENDERBUFFER,Ae,Ee,v.width,v.height):n.renderbufferStorage(n.RENDERBUFFER,Ee,v.width,v.height)}}n.bindRenderbuffer(n.RENDERBUFFER,null)}function Te(A,v){if(v&&v.isWebGLCubeRenderTarget)throw new Error("Depth Texture with cube render targets is not supported");if(t.bindFramebuffer(n.FRAMEBUFFER,A),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const K=i.get(v.depthTexture);K.__renderTarget=v,(!K.__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),X(v.depthTexture,0);const ne=K.__webglTexture,Z=vt(v);if(v.depthTexture.format===Gs)we(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,n.DEPTH_ATTACHMENT,n.TEXTURE_2D,ne,0,Z):n.framebufferTexture2D(n.FRAMEBUFFER,n.DEPTH_ATTACHMENT,n.TEXTURE_2D,ne,0);else if(v.depthTexture.format===Vs)we(v)?a.framebufferTexture2DMultisampleEXT(n.FRAMEBUFFER,n.DEPTH_STENCIL_ATTACHMENT,n.TEXTURE_2D,ne,0,Z):n.framebufferTexture2D(n.FRAMEBUFFER,n.DEPTH_STENCIL_ATTACHMENT,n.TEXTURE_2D,ne,0);else throw new Error("Unknown depthTexture format")}function Je(A){const v=i.get(A),B=A.isWebGLCubeRenderTarget===!0;if(v.__boundDepthTexture!==A.depthTexture){const K=A.depthTexture;if(v.__depthDisposeCallback&&v.__depthDisposeCallback(),K){const ne=()=>{delete v.__boundDepthTexture,delete v.__depthDisposeCallback,K.removeEventListener("dispose",ne)};K.addEventListener("dispose",ne),v.__depthDisposeCallback=ne}v.__boundDepthTexture=K}if(A.depthTexture&&!v.__autoAllocateDepthBuffer){if(B)throw new Error("target.depthTexture not supported in Cube render targets");const K=A.texture.mipmaps;K&&K.length>0?Te(v.__webglFramebuffer[0],A):Te(v.__webglFramebuffer,A)}else if(B){v.__webglDepthbuffer=[];for(let K=0;K<6;K++)if(t.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer[K]),v.__webglDepthbuffer[K]===void 0)v.__webglDepthbuffer[K]=n.createRenderbuffer(),Fe(v.__webglDepthbuffer[K],A,!1);else{const ne=A.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,Z=v.__webglDepthbuffer[K];n.bindRenderbuffer(n.RENDERBUFFER,Z),n.framebufferRenderbuffer(n.FRAMEBUFFER,ne,n.RENDERBUFFER,Z)}}else{const K=A.texture.mipmaps;if(K&&K.length>0?t.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer[0]):t.bindFramebuffer(n.FRAMEBUFFER,v.__webglFramebuffer),v.__webglDepthbuffer===void 0)v.__webglDepthbuffer=n.createRenderbuffer(),Fe(v.__webglDepthbuffer,A,!1);else{const ne=A.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,Z=v.__webglDepthbuffer;n.bindRenderbuffer(n.RENDERBUFFER,Z),n.framebufferRenderbuffer(n.FRAMEBUFFER,ne,n.RENDERBUFFER,Z)}}t.bindFramebuffer(n.FRAMEBUFFER,null)}function kt(A,v,B){const K=i.get(A);v!==void 0&&xe(K.__webglFramebuffer,A,A.texture,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,0),B!==void 0&&Je(A)}function I(A){const v=A.texture,B=i.get(A),K=i.get(v);A.addEventListener("dispose",T);const ne=A.textures,Z=A.isWebGLCubeRenderTarget===!0,Re=ne.length>1;if(Re||(K.__webglTexture===void 0&&(K.__webglTexture=n.createTexture()),K.__version=v.version,o.memory.textures++),Z){B.__webglFramebuffer=[];for(let ce=0;ce<6;ce++)if(v.mipmaps&&v.mipmaps.length>0){B.__webglFramebuffer[ce]=[];for(let Ee=0;Ee<v.mipmaps.length;Ee++)B.__webglFramebuffer[ce][Ee]=n.createFramebuffer()}else B.__webglFramebuffer[ce]=n.createFramebuffer()}else{if(v.mipmaps&&v.mipmaps.length>0){B.__webglFramebuffer=[];for(let ce=0;ce<v.mipmaps.length;ce++)B.__webglFramebuffer[ce]=n.createFramebuffer()}else B.__webglFramebuffer=n.createFramebuffer();if(Re)for(let ce=0,Ee=ne.length;ce<Ee;ce++){const Ae=i.get(ne[ce]);Ae.__webglTexture===void 0&&(Ae.__webglTexture=n.createTexture(),o.memory.textures++)}if(A.samples>0&&we(A)===!1){B.__webglMultisampledFramebuffer=n.createFramebuffer(),B.__webglColorRenderbuffer=[],t.bindFramebuffer(n.FRAMEBUFFER,B.__webglMultisampledFramebuffer);for(let ce=0;ce<ne.length;ce++){const Ee=ne[ce];B.__webglColorRenderbuffer[ce]=n.createRenderbuffer(),n.bindRenderbuffer(n.RENDERBUFFER,B.__webglColorRenderbuffer[ce]);const Ae=r.convert(Ee.format,Ee.colorSpace),re=r.convert(Ee.type),ge=M(Ee.internalFormat,Ae,re,Ee.colorSpace,A.isXRRenderTarget===!0),Oe=vt(A);n.renderbufferStorageMultisample(n.RENDERBUFFER,Oe,ge,A.width,A.height),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+ce,n.RENDERBUFFER,B.__webglColorRenderbuffer[ce])}n.bindRenderbuffer(n.RENDERBUFFER,null),A.depthBuffer&&(B.__webglDepthRenderbuffer=n.createRenderbuffer(),Fe(B.__webglDepthRenderbuffer,A,!0)),t.bindFramebuffer(n.FRAMEBUFFER,null)}}if(Z){t.bindTexture(n.TEXTURE_CUBE_MAP,K.__webglTexture),Q(n.TEXTURE_CUBE_MAP,v);for(let ce=0;ce<6;ce++)if(v.mipmaps&&v.mipmaps.length>0)for(let Ee=0;Ee<v.mipmaps.length;Ee++)xe(B.__webglFramebuffer[ce][Ee],A,v,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+ce,Ee);else xe(B.__webglFramebuffer[ce],A,v,n.COLOR_ATTACHMENT0,n.TEXTURE_CUBE_MAP_POSITIVE_X+ce,0);m(v)&&p(n.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Re){for(let ce=0,Ee=ne.length;ce<Ee;ce++){const Ae=ne[ce],re=i.get(Ae);let ge=n.TEXTURE_2D;(A.isWebGL3DRenderTarget||A.isWebGLArrayRenderTarget)&&(ge=A.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),t.bindTexture(ge,re.__webglTexture),Q(ge,Ae),xe(B.__webglFramebuffer,A,Ae,n.COLOR_ATTACHMENT0+ce,ge,0),m(Ae)&&p(ge)}t.unbindTexture()}else{let ce=n.TEXTURE_2D;if((A.isWebGL3DRenderTarget||A.isWebGLArrayRenderTarget)&&(ce=A.isWebGL3DRenderTarget?n.TEXTURE_3D:n.TEXTURE_2D_ARRAY),t.bindTexture(ce,K.__webglTexture),Q(ce,v),v.mipmaps&&v.mipmaps.length>0)for(let Ee=0;Ee<v.mipmaps.length;Ee++)xe(B.__webglFramebuffer[Ee],A,v,n.COLOR_ATTACHMENT0,ce,Ee);else xe(B.__webglFramebuffer,A,v,n.COLOR_ATTACHMENT0,ce,0);m(v)&&p(ce),t.unbindTexture()}A.depthBuffer&&Je(A)}function _t(A){const v=A.textures;for(let B=0,K=v.length;B<K;B++){const ne=v[B];if(m(ne)){const Z=C(A),Re=i.get(ne).__webglTexture;t.bindTexture(Z,Re),p(Z),t.unbindTexture()}}}const He=[],ke=[];function Me(A){if(A.samples>0){if(we(A)===!1){const v=A.textures,B=A.width,K=A.height;let ne=n.COLOR_BUFFER_BIT;const Z=A.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT,Re=i.get(A),ce=v.length>1;if(ce)for(let Ae=0;Ae<v.length;Ae++)t.bindFramebuffer(n.FRAMEBUFFER,Re.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+Ae,n.RENDERBUFFER,null),t.bindFramebuffer(n.FRAMEBUFFER,Re.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+Ae,n.TEXTURE_2D,null,0);t.bindFramebuffer(n.READ_FRAMEBUFFER,Re.__webglMultisampledFramebuffer);const Ee=A.texture.mipmaps;Ee&&Ee.length>0?t.bindFramebuffer(n.DRAW_FRAMEBUFFER,Re.__webglFramebuffer[0]):t.bindFramebuffer(n.DRAW_FRAMEBUFFER,Re.__webglFramebuffer);for(let Ae=0;Ae<v.length;Ae++){if(A.resolveDepthBuffer&&(A.depthBuffer&&(ne|=n.DEPTH_BUFFER_BIT),A.stencilBuffer&&A.resolveStencilBuffer&&(ne|=n.STENCIL_BUFFER_BIT)),ce){n.framebufferRenderbuffer(n.READ_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.RENDERBUFFER,Re.__webglColorRenderbuffer[Ae]);const re=i.get(v[Ae]).__webglTexture;n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,re,0)}n.blitFramebuffer(0,0,B,K,0,0,B,K,ne,n.NEAREST),l===!0&&(He.length=0,ke.length=0,He.push(n.COLOR_ATTACHMENT0+Ae),A.depthBuffer&&A.resolveDepthBuffer===!1&&(He.push(Z),ke.push(Z),n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,ke)),n.invalidateFramebuffer(n.READ_FRAMEBUFFER,He))}if(t.bindFramebuffer(n.READ_FRAMEBUFFER,null),t.bindFramebuffer(n.DRAW_FRAMEBUFFER,null),ce)for(let Ae=0;Ae<v.length;Ae++){t.bindFramebuffer(n.FRAMEBUFFER,Re.__webglMultisampledFramebuffer),n.framebufferRenderbuffer(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0+Ae,n.RENDERBUFFER,Re.__webglColorRenderbuffer[Ae]);const re=i.get(v[Ae]).__webglTexture;t.bindFramebuffer(n.FRAMEBUFFER,Re.__webglFramebuffer),n.framebufferTexture2D(n.DRAW_FRAMEBUFFER,n.COLOR_ATTACHMENT0+Ae,n.TEXTURE_2D,re,0)}t.bindFramebuffer(n.DRAW_FRAMEBUFFER,Re.__webglMultisampledFramebuffer)}else if(A.depthBuffer&&A.resolveDepthBuffer===!1&&l){const v=A.stencilBuffer?n.DEPTH_STENCIL_ATTACHMENT:n.DEPTH_ATTACHMENT;n.invalidateFramebuffer(n.DRAW_FRAMEBUFFER,[v])}}}function vt(A){return Math.min(s.maxSamples,A.samples)}function we(A){const v=i.get(A);return A.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function Xe(A){const v=o.render.frame;u.get(A)!==v&&(u.set(A,v),A.update())}function Nt(A,v){const B=A.colorSpace,K=A.format,ne=A.type;return A.isCompressedTexture===!0||A.isVideoTexture===!0||B!==us&&B!==ti&&(tt.getTransfer(B)===rt?(K!==vn||ne!==Rn)&&console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):console.error("THREE.WebGLTextures: Unsupported texture color space:",B)),v}function wt(A){return typeof HTMLImageElement<"u"&&A instanceof HTMLImageElement?(c.width=A.naturalWidth||A.width,c.height=A.naturalHeight||A.height):typeof VideoFrame<"u"&&A instanceof VideoFrame?(c.width=A.displayWidth,c.height=A.displayHeight):(c.width=A.width,c.height=A.height),c}this.allocateTextureUnit=G,this.resetTextureUnits=O,this.setTexture2D=X,this.setTexture2DArray=q,this.setTexture3D=J,this.setTextureCube=z,this.rebindTextures=kt,this.setupRenderTarget=I,this.updateRenderTargetMipmap=_t,this.updateMultisampleRenderTarget=Me,this.setupDepthRenderbuffer=Je,this.setupFrameBufferTexture=xe,this.useMultisampledRTT=we}function ZS(n,e){function t(i,s=ti){let r;const o=tt.getTransfer(s);if(i===Rn)return n.UNSIGNED_BYTE;if(i===Uc)return n.UNSIGNED_SHORT_4_4_4_4;if(i===Oc)return n.UNSIGNED_SHORT_5_5_5_1;if(i===Xh)return n.UNSIGNED_INT_5_9_9_9_REV;if(i===$h)return n.UNSIGNED_INT_10F_11F_11F_REV;if(i===Vh)return n.BYTE;if(i===Wh)return n.SHORT;if(i===Bs)return n.UNSIGNED_SHORT;if(i===Nc)return n.INT;if(i===Ei)return n.UNSIGNED_INT;if(i===Wn)return n.FLOAT;if(i===Ks)return n.HALF_FLOAT;if(i===Yh)return n.ALPHA;if(i===qh)return n.RGB;if(i===vn)return n.RGBA;if(i===Gs)return n.DEPTH_COMPONENT;if(i===Vs)return n.DEPTH_STENCIL;if(i===jh)return n.RED;if(i===Fc)return n.RED_INTEGER;if(i===Zh)return n.RG;if(i===kc)return n.RG_INTEGER;if(i===zc)return n.RGBA_INTEGER;if(i===qr||i===jr||i===Zr||i===Jr)if(o===rt)if(r=e.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(i===qr)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(i===jr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(i===Zr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(i===Jr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=e.get("WEBGL_compressed_texture_s3tc"),r!==null){if(i===qr)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(i===jr)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(i===Zr)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(i===Jr)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(i===Wa||i===Xa||i===$a||i===Ya)if(r=e.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(i===Wa)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(i===Xa)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(i===$a)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(i===Ya)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(i===qa||i===ja||i===Za)if(r=e.get("WEBGL_compressed_texture_etc"),r!==null){if(i===qa||i===ja)return o===rt?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(i===Za)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC}else return null;if(i===Ja||i===Ka||i===Qa||i===ec||i===tc||i===nc||i===ic||i===sc||i===rc||i===oc||i===ac||i===cc||i===lc||i===uc)if(r=e.get("WEBGL_compressed_texture_astc"),r!==null){if(i===Ja)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(i===Ka)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(i===Qa)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(i===ec)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(i===tc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(i===nc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(i===ic)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(i===sc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(i===rc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(i===oc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(i===ac)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(i===cc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(i===lc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(i===uc)return o===rt?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(i===hc||i===dc||i===fc)if(r=e.get("EXT_texture_compression_bptc"),r!==null){if(i===hc)return o===rt?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(i===dc)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(i===fc)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(i===pc||i===mc||i===gc||i===_c)if(r=e.get("EXT_texture_compression_rgtc"),r!==null){if(i===pc)return r.COMPRESSED_RED_RGTC1_EXT;if(i===mc)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(i===gc)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(i===_c)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return i===Hs?n.UNSIGNED_INT_24_8:n[i]!==void 0?n[i]:null}return{convert:t}}const JS=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,KS=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class QS{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const i=new ud(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=i}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,i=new ai({vertexShader:JS,fragmentShader:KS,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new xt(new ms(20,20),i)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class eb extends Ri{constructor(e,t){super();const i=this;let s=null,r=1,o=null,a="local-floor",l=1,c=null,u=null,h=null,d=null,f=null,g=null;const _=typeof XRWebGLBinding<"u",m=new QS,p={},C=t.getContextAttributes();let M=null,x=null;const P=[],w=[],T=new ue;let L=null;const y=new sn;y.viewport=new ct;const b=new sn;b.viewport=new ct;const N=[y,b],O=new vv;let G=null,V=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(j){let te=P[j];return te===void 0&&(te=new ta,P[j]=te),te.getTargetRaySpace()},this.getControllerGrip=function(j){let te=P[j];return te===void 0&&(te=new ta,P[j]=te),te.getGripSpace()},this.getHand=function(j){let te=P[j];return te===void 0&&(te=new ta,P[j]=te),te.getHandSpace()};function X(j){const te=w.indexOf(j.inputSource);if(te===-1)return;const xe=P[te];xe!==void 0&&(xe.update(j.inputSource,j.frame,c||o),xe.dispatchEvent({type:j.type,data:j.inputSource}))}function q(){s.removeEventListener("select",X),s.removeEventListener("selectstart",X),s.removeEventListener("selectend",X),s.removeEventListener("squeeze",X),s.removeEventListener("squeezestart",X),s.removeEventListener("squeezeend",X),s.removeEventListener("end",q),s.removeEventListener("inputsourceschange",J);for(let j=0;j<P.length;j++){const te=w[j];te!==null&&(w[j]=null,P[j].disconnect(te))}G=null,V=null,m.reset();for(const j in p)delete p[j];e.setRenderTarget(M),f=null,d=null,h=null,s=null,x=null,Ie.stop(),i.isPresenting=!1,e.setPixelRatio(L),e.setSize(T.width,T.height,!1),i.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(j){r=j,i.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(j){a=j,i.isPresenting===!0&&console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return c||o},this.setReferenceSpace=function(j){c=j},this.getBaseLayer=function(){return d!==null?d:f},this.getBinding=function(){return h===null&&_&&(h=new XRWebGLBinding(s,t)),h},this.getFrame=function(){return g},this.getSession=function(){return s},this.setSession=async function(j){if(s=j,s!==null){if(M=e.getRenderTarget(),s.addEventListener("select",X),s.addEventListener("selectstart",X),s.addEventListener("selectend",X),s.addEventListener("squeeze",X),s.addEventListener("squeezestart",X),s.addEventListener("squeezeend",X),s.addEventListener("end",q),s.addEventListener("inputsourceschange",J),C.xrCompatible!==!0&&await t.makeXRCompatible(),L=e.getPixelRatio(),e.getSize(T),_&&"createProjectionLayer"in XRWebGLBinding.prototype){let xe=null,Fe=null,Te=null;C.depth&&(Te=C.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,xe=C.stencil?Vs:Gs,Fe=C.stencil?Hs:Ei);const Je={colorFormat:t.RGBA8,depthFormat:Te,scaleFactor:r};h=this.getBinding(),d=h.createProjectionLayer(Je),s.updateRenderState({layers:[d]}),e.setPixelRatio(1),e.setSize(d.textureWidth,d.textureHeight,!1),x=new Ai(d.textureWidth,d.textureHeight,{format:vn,type:Rn,depthTexture:new ld(d.textureWidth,d.textureHeight,Fe,void 0,void 0,void 0,void 0,void 0,void 0,xe),stencilBuffer:C.stencil,colorSpace:e.outputColorSpace,samples:C.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1})}else{const xe={antialias:C.antialias,alpha:!0,depth:C.depth,stencil:C.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(s,t,xe),s.updateRenderState({baseLayer:f}),e.setPixelRatio(1),e.setSize(f.framebufferWidth,f.framebufferHeight,!1),x=new Ai(f.framebufferWidth,f.framebufferHeight,{format:vn,type:Rn,colorSpace:e.outputColorSpace,stencilBuffer:C.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}x.isXRRenderTarget=!0,this.setFoveation(l),c=null,o=await s.requestReferenceSpace(a),Ie.setContext(s),Ie.start(),i.isPresenting=!0,i.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return m.getDepthTexture()};function J(j){for(let te=0;te<j.removed.length;te++){const xe=j.removed[te],Fe=w.indexOf(xe);Fe>=0&&(w[Fe]=null,P[Fe].disconnect(xe))}for(let te=0;te<j.added.length;te++){const xe=j.added[te];let Fe=w.indexOf(xe);if(Fe===-1){for(let Je=0;Je<P.length;Je++)if(Je>=w.length){w.push(xe),Fe=Je;break}else if(w[Je]===null){w[Je]=xe,Fe=Je;break}if(Fe===-1)break}const Te=P[Fe];Te&&Te.connect(xe)}}const z=new R,H=new R;function D(j,te,xe){z.setFromMatrixPosition(te.matrixWorld),H.setFromMatrixPosition(xe.matrixWorld);const Fe=z.distanceTo(H),Te=te.projectionMatrix.elements,Je=xe.projectionMatrix.elements,kt=Te[14]/(Te[10]-1),I=Te[14]/(Te[10]+1),_t=(Te[9]+1)/Te[5],He=(Te[9]-1)/Te[5],ke=(Te[8]-1)/Te[0],Me=(Je[8]+1)/Je[0],vt=kt*ke,we=kt*Me,Xe=Fe/(-ke+Me),Nt=Xe*-ke;if(te.matrixWorld.decompose(j.position,j.quaternion,j.scale),j.translateX(Nt),j.translateZ(Xe),j.matrixWorld.compose(j.position,j.quaternion,j.scale),j.matrixWorldInverse.copy(j.matrixWorld).invert(),Te[10]===-1)j.projectionMatrix.copy(te.projectionMatrix),j.projectionMatrixInverse.copy(te.projectionMatrixInverse);else{const wt=kt+Xe,A=I+Xe,v=vt-Nt,B=we+(Fe-Nt),K=_t*I/A*wt,ne=He*I/A*wt;j.projectionMatrix.makePerspective(v,B,K,ne,wt,A),j.projectionMatrixInverse.copy(j.projectionMatrix).invert()}}function W(j,te){te===null?j.matrixWorld.copy(j.matrix):j.matrixWorld.multiplyMatrices(te.matrixWorld,j.matrix),j.matrixWorldInverse.copy(j.matrixWorld).invert()}this.updateCamera=function(j){if(s===null)return;let te=j.near,xe=j.far;m.texture!==null&&(m.depthNear>0&&(te=m.depthNear),m.depthFar>0&&(xe=m.depthFar)),O.near=b.near=y.near=te,O.far=b.far=y.far=xe,(G!==O.near||V!==O.far)&&(s.updateRenderState({depthNear:O.near,depthFar:O.far}),G=O.near,V=O.far),O.layers.mask=j.layers.mask|6,y.layers.mask=O.layers.mask&3,b.layers.mask=O.layers.mask&5;const Fe=j.parent,Te=O.cameras;W(O,Fe);for(let Je=0;Je<Te.length;Je++)W(Te[Je],Fe);Te.length===2?D(O,y,b):O.projectionMatrix.copy(y.projectionMatrix),Q(j,O,Fe)};function Q(j,te,xe){xe===null?j.matrix.copy(te.matrixWorld):(j.matrix.copy(xe.matrixWorld),j.matrix.invert(),j.matrix.multiply(te.matrixWorld)),j.matrix.decompose(j.position,j.quaternion,j.scale),j.updateMatrixWorld(!0),j.projectionMatrix.copy(te.projectionMatrix),j.projectionMatrixInverse.copy(te.projectionMatrixInverse),j.isPerspectiveCamera&&(j.fov=xc*2*Math.atan(1/j.projectionMatrix.elements[5]),j.zoom=1)}this.getCamera=function(){return O},this.getFoveation=function(){if(!(d===null&&f===null))return l},this.setFoveation=function(j){l=j,d!==null&&(d.fixedFoveation=j),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=j)},this.hasDepthSensing=function(){return m.texture!==null},this.getDepthSensingMesh=function(){return m.getMesh(O)},this.getCameraTexture=function(j){return p[j]};let me=null;function De(j,te){if(u=te.getViewerPose(c||o),g=te,u!==null){const xe=u.views;f!==null&&(e.setRenderTargetFramebuffer(x,f.framebuffer),e.setRenderTarget(x));let Fe=!1;xe.length!==O.cameras.length&&(O.cameras.length=0,Fe=!0);for(let I=0;I<xe.length;I++){const _t=xe[I];let He=null;if(f!==null)He=f.getViewport(_t);else{const Me=h.getViewSubImage(d,_t);He=Me.viewport,I===0&&(e.setRenderTargetTextures(x,Me.colorTexture,Me.depthStencilTexture),e.setRenderTarget(x))}let ke=N[I];ke===void 0&&(ke=new sn,ke.layers.enable(I),ke.viewport=new ct,N[I]=ke),ke.matrix.fromArray(_t.transform.matrix),ke.matrix.decompose(ke.position,ke.quaternion,ke.scale),ke.projectionMatrix.fromArray(_t.projectionMatrix),ke.projectionMatrixInverse.copy(ke.projectionMatrix).invert(),ke.viewport.set(He.x,He.y,He.width,He.height),I===0&&(O.matrix.copy(ke.matrix),O.matrix.decompose(O.position,O.quaternion,O.scale)),Fe===!0&&O.cameras.push(ke)}const Te=s.enabledFeatures;if(Te&&Te.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&_){h=i.getBinding();const I=h.getDepthInformation(xe[0]);I&&I.isValid&&I.texture&&m.init(I,s.renderState)}if(Te&&Te.includes("camera-access")&&_){e.state.unbindTexture(),h=i.getBinding();for(let I=0;I<xe.length;I++){const _t=xe[I].camera;if(_t){let He=p[_t];He||(He=new ud,p[_t]=He);const ke=h.getCameraImage(_t);He.sourceTexture=ke}}}}for(let xe=0;xe<P.length;xe++){const Fe=w[xe],Te=P[xe];Fe!==null&&Te!==void 0&&Te.update(Fe,te,c||o)}me&&me(j,te),te.detectedPlanes&&i.dispatchEvent({type:"planesdetected",data:te}),g=null}const Ie=new Sd;Ie.setAnimationLoop(De),this.setAnimationLoop=function(j){me=j},this.dispose=function(){}}}const gi=new Yn,tb=new ft;function nb(n,e){function t(m,p){m.matrixAutoUpdate===!0&&m.updateMatrix(),p.value.copy(m.matrix)}function i(m,p){p.color.getRGB(m.fogColor.value,id(n)),p.isFog?(m.fogNear.value=p.near,m.fogFar.value=p.far):p.isFogExp2&&(m.fogDensity.value=p.density)}function s(m,p,C,M,x){p.isMeshBasicMaterial||p.isMeshLambertMaterial?r(m,p):p.isMeshToonMaterial?(r(m,p),h(m,p)):p.isMeshPhongMaterial?(r(m,p),u(m,p)):p.isMeshStandardMaterial?(r(m,p),d(m,p),p.isMeshPhysicalMaterial&&f(m,p,x)):p.isMeshMatcapMaterial?(r(m,p),g(m,p)):p.isMeshDepthMaterial?r(m,p):p.isMeshDistanceMaterial?(r(m,p),_(m,p)):p.isMeshNormalMaterial?r(m,p):p.isLineBasicMaterial?(o(m,p),p.isLineDashedMaterial&&a(m,p)):p.isPointsMaterial?l(m,p,C,M):p.isSpriteMaterial?c(m,p):p.isShadowMaterial?(m.color.value.copy(p.color),m.opacity.value=p.opacity):p.isShaderMaterial&&(p.uniformsNeedUpdate=!1)}function r(m,p){m.opacity.value=p.opacity,p.color&&m.diffuse.value.copy(p.color),p.emissive&&m.emissive.value.copy(p.emissive).multiplyScalar(p.emissiveIntensity),p.map&&(m.map.value=p.map,t(p.map,m.mapTransform)),p.alphaMap&&(m.alphaMap.value=p.alphaMap,t(p.alphaMap,m.alphaMapTransform)),p.bumpMap&&(m.bumpMap.value=p.bumpMap,t(p.bumpMap,m.bumpMapTransform),m.bumpScale.value=p.bumpScale,p.side===Jt&&(m.bumpScale.value*=-1)),p.normalMap&&(m.normalMap.value=p.normalMap,t(p.normalMap,m.normalMapTransform),m.normalScale.value.copy(p.normalScale),p.side===Jt&&m.normalScale.value.negate()),p.displacementMap&&(m.displacementMap.value=p.displacementMap,t(p.displacementMap,m.displacementMapTransform),m.displacementScale.value=p.displacementScale,m.displacementBias.value=p.displacementBias),p.emissiveMap&&(m.emissiveMap.value=p.emissiveMap,t(p.emissiveMap,m.emissiveMapTransform)),p.specularMap&&(m.specularMap.value=p.specularMap,t(p.specularMap,m.specularMapTransform)),p.alphaTest>0&&(m.alphaTest.value=p.alphaTest);const C=e.get(p),M=C.envMap,x=C.envMapRotation;M&&(m.envMap.value=M,gi.copy(x),gi.x*=-1,gi.y*=-1,gi.z*=-1,M.isCubeTexture&&M.isRenderTargetTexture===!1&&(gi.y*=-1,gi.z*=-1),m.envMapRotation.value.setFromMatrix4(tb.makeRotationFromEuler(gi)),m.flipEnvMap.value=M.isCubeTexture&&M.isRenderTargetTexture===!1?-1:1,m.reflectivity.value=p.reflectivity,m.ior.value=p.ior,m.refractionRatio.value=p.refractionRatio),p.lightMap&&(m.lightMap.value=p.lightMap,m.lightMapIntensity.value=p.lightMapIntensity,t(p.lightMap,m.lightMapTransform)),p.aoMap&&(m.aoMap.value=p.aoMap,m.aoMapIntensity.value=p.aoMapIntensity,t(p.aoMap,m.aoMapTransform))}function o(m,p){m.diffuse.value.copy(p.color),m.opacity.value=p.opacity,p.map&&(m.map.value=p.map,t(p.map,m.mapTransform))}function a(m,p){m.dashSize.value=p.dashSize,m.totalSize.value=p.dashSize+p.gapSize,m.scale.value=p.scale}function l(m,p,C,M){m.diffuse.value.copy(p.color),m.opacity.value=p.opacity,m.size.value=p.size*C,m.scale.value=M*.5,p.map&&(m.map.value=p.map,t(p.map,m.uvTransform)),p.alphaMap&&(m.alphaMap.value=p.alphaMap,t(p.alphaMap,m.alphaMapTransform)),p.alphaTest>0&&(m.alphaTest.value=p.alphaTest)}function c(m,p){m.diffuse.value.copy(p.color),m.opacity.value=p.opacity,m.rotation.value=p.rotation,p.map&&(m.map.value=p.map,t(p.map,m.mapTransform)),p.alphaMap&&(m.alphaMap.value=p.alphaMap,t(p.alphaMap,m.alphaMapTransform)),p.alphaTest>0&&(m.alphaTest.value=p.alphaTest)}function u(m,p){m.specular.value.copy(p.specular),m.shininess.value=Math.max(p.shininess,1e-4)}function h(m,p){p.gradientMap&&(m.gradientMap.value=p.gradientMap)}function d(m,p){m.metalness.value=p.metalness,p.metalnessMap&&(m.metalnessMap.value=p.metalnessMap,t(p.metalnessMap,m.metalnessMapTransform)),m.roughness.value=p.roughness,p.roughnessMap&&(m.roughnessMap.value=p.roughnessMap,t(p.roughnessMap,m.roughnessMapTransform)),p.envMap&&(m.envMapIntensity.value=p.envMapIntensity)}function f(m,p,C){m.ior.value=p.ior,p.sheen>0&&(m.sheenColor.value.copy(p.sheenColor).multiplyScalar(p.sheen),m.sheenRoughness.value=p.sheenRoughness,p.sheenColorMap&&(m.sheenColorMap.value=p.sheenColorMap,t(p.sheenColorMap,m.sheenColorMapTransform)),p.sheenRoughnessMap&&(m.sheenRoughnessMap.value=p.sheenRoughnessMap,t(p.sheenRoughnessMap,m.sheenRoughnessMapTransform))),p.clearcoat>0&&(m.clearcoat.value=p.clearcoat,m.clearcoatRoughness.value=p.clearcoatRoughness,p.clearcoatMap&&(m.clearcoatMap.value=p.clearcoatMap,t(p.clearcoatMap,m.clearcoatMapTransform)),p.clearcoatRoughnessMap&&(m.clearcoatRoughnessMap.value=p.clearcoatRoughnessMap,t(p.clearcoatRoughnessMap,m.clearcoatRoughnessMapTransform)),p.clearcoatNormalMap&&(m.clearcoatNormalMap.value=p.clearcoatNormalMap,t(p.clearcoatNormalMap,m.clearcoatNormalMapTransform),m.clearcoatNormalScale.value.copy(p.clearcoatNormalScale),p.side===Jt&&m.clearcoatNormalScale.value.negate())),p.dispersion>0&&(m.dispersion.value=p.dispersion),p.iridescence>0&&(m.iridescence.value=p.iridescence,m.iridescenceIOR.value=p.iridescenceIOR,m.iridescenceThicknessMinimum.value=p.iridescenceThicknessRange[0],m.iridescenceThicknessMaximum.value=p.iridescenceThicknessRange[1],p.iridescenceMap&&(m.iridescenceMap.value=p.iridescenceMap,t(p.iridescenceMap,m.iridescenceMapTransform)),p.iridescenceThicknessMap&&(m.iridescenceThicknessMap.value=p.iridescenceThicknessMap,t(p.iridescenceThicknessMap,m.iridescenceThicknessMapTransform))),p.transmission>0&&(m.transmission.value=p.transmission,m.transmissionSamplerMap.value=C.texture,m.transmissionSamplerSize.value.set(C.width,C.height),p.transmissionMap&&(m.transmissionMap.value=p.transmissionMap,t(p.transmissionMap,m.transmissionMapTransform)),m.thickness.value=p.thickness,p.thicknessMap&&(m.thicknessMap.value=p.thicknessMap,t(p.thicknessMap,m.thicknessMapTransform)),m.attenuationDistance.value=p.attenuationDistance,m.attenuationColor.value.copy(p.attenuationColor)),p.anisotropy>0&&(m.anisotropyVector.value.set(p.anisotropy*Math.cos(p.anisotropyRotation),p.anisotropy*Math.sin(p.anisotropyRotation)),p.anisotropyMap&&(m.anisotropyMap.value=p.anisotropyMap,t(p.anisotropyMap,m.anisotropyMapTransform))),m.specularIntensity.value=p.specularIntensity,m.specularColor.value.copy(p.specularColor),p.specularColorMap&&(m.specularColorMap.value=p.specularColorMap,t(p.specularColorMap,m.specularColorMapTransform)),p.specularIntensityMap&&(m.specularIntensityMap.value=p.specularIntensityMap,t(p.specularIntensityMap,m.specularIntensityMapTransform))}function g(m,p){p.matcap&&(m.matcap.value=p.matcap)}function _(m,p){const C=e.get(p).light;m.referencePosition.value.setFromMatrixPosition(C.matrixWorld),m.nearDistance.value=C.shadow.camera.near,m.farDistance.value=C.shadow.camera.far}return{refreshFogUniforms:i,refreshMaterialUniforms:s}}function ib(n,e,t,i){let s={},r={},o=[];const a=n.getParameter(n.MAX_UNIFORM_BUFFER_BINDINGS);function l(C,M){const x=M.program;i.uniformBlockBinding(C,x)}function c(C,M){let x=s[C.id];x===void 0&&(g(C),x=u(C),s[C.id]=x,C.addEventListener("dispose",m));const P=M.program;i.updateUBOMapping(C,P);const w=e.render.frame;r[C.id]!==w&&(d(C),r[C.id]=w)}function u(C){const M=h();C.__bindingPointIndex=M;const x=n.createBuffer(),P=C.__size,w=C.usage;return n.bindBuffer(n.UNIFORM_BUFFER,x),n.bufferData(n.UNIFORM_BUFFER,P,w),n.bindBuffer(n.UNIFORM_BUFFER,null),n.bindBufferBase(n.UNIFORM_BUFFER,M,x),x}function h(){for(let C=0;C<a;C++)if(o.indexOf(C)===-1)return o.push(C),C;return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function d(C){const M=s[C.id],x=C.uniforms,P=C.__cache;n.bindBuffer(n.UNIFORM_BUFFER,M);for(let w=0,T=x.length;w<T;w++){const L=Array.isArray(x[w])?x[w]:[x[w]];for(let y=0,b=L.length;y<b;y++){const N=L[y];if(f(N,w,y,P)===!0){const O=N.__offset,G=Array.isArray(N.value)?N.value:[N.value];let V=0;for(let X=0;X<G.length;X++){const q=G[X],J=_(q);typeof q=="number"||typeof q=="boolean"?(N.__data[0]=q,n.bufferSubData(n.UNIFORM_BUFFER,O+V,N.__data)):q.isMatrix3?(N.__data[0]=q.elements[0],N.__data[1]=q.elements[1],N.__data[2]=q.elements[2],N.__data[3]=0,N.__data[4]=q.elements[3],N.__data[5]=q.elements[4],N.__data[6]=q.elements[5],N.__data[7]=0,N.__data[8]=q.elements[6],N.__data[9]=q.elements[7],N.__data[10]=q.elements[8],N.__data[11]=0):(q.toArray(N.__data,V),V+=J.storage/Float32Array.BYTES_PER_ELEMENT)}n.bufferSubData(n.UNIFORM_BUFFER,O,N.__data)}}}n.bindBuffer(n.UNIFORM_BUFFER,null)}function f(C,M,x,P){const w=C.value,T=M+"_"+x;if(P[T]===void 0)return typeof w=="number"||typeof w=="boolean"?P[T]=w:P[T]=w.clone(),!0;{const L=P[T];if(typeof w=="number"||typeof w=="boolean"){if(L!==w)return P[T]=w,!0}else if(L.equals(w)===!1)return L.copy(w),!0}return!1}function g(C){const M=C.uniforms;let x=0;const P=16;for(let T=0,L=M.length;T<L;T++){const y=Array.isArray(M[T])?M[T]:[M[T]];for(let b=0,N=y.length;b<N;b++){const O=y[b],G=Array.isArray(O.value)?O.value:[O.value];for(let V=0,X=G.length;V<X;V++){const q=G[V],J=_(q),z=x%P,H=z%J.boundary,D=z+H;x+=H,D!==0&&P-D<J.storage&&(x+=P-D),O.__data=new Float32Array(J.storage/Float32Array.BYTES_PER_ELEMENT),O.__offset=x,x+=J.storage}}}const w=x%P;return w>0&&(x+=P-w),C.__size=x,C.__cache={},this}function _(C){const M={boundary:0,storage:0};return typeof C=="number"||typeof C=="boolean"?(M.boundary=4,M.storage=4):C.isVector2?(M.boundary=8,M.storage=8):C.isVector3||C.isColor?(M.boundary=16,M.storage=12):C.isVector4?(M.boundary=16,M.storage=16):C.isMatrix3?(M.boundary=48,M.storage=48):C.isMatrix4?(M.boundary=64,M.storage=64):C.isTexture?console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group."):console.warn("THREE.WebGLRenderer: Unsupported uniform value type.",C),M}function m(C){const M=C.target;M.removeEventListener("dispose",m);const x=o.indexOf(M.__bindingPointIndex);o.splice(x,1),n.deleteBuffer(s[M.id]),delete s[M.id],delete r[M.id]}function p(){for(const C in s)n.deleteBuffer(s[C]);o=[],s={},r={}}return{bind:l,update:c,dispose:p}}class sb{constructor(e={}){const{canvas:t=Qg(),context:i=null,depth:s=!0,stencil:r=!1,alpha:o=!1,antialias:a=!1,premultipliedAlpha:l=!0,preserveDrawingBuffer:c=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:h=!1,reversedDepthBuffer:d=!1}=e;this.isWebGLRenderer=!0;let f;if(i!==null){if(typeof WebGLRenderingContext<"u"&&i instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");f=i.getContextAttributes().alpha}else f=o;const g=new Uint32Array(4),_=new Int32Array(4);let m=null,p=null;const C=[],M=[];this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=ri,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const x=this;let P=!1;this._outputColorSpace=Ot;let w=0,T=0,L=null,y=-1,b=null;const N=new ct,O=new ct;let G=null;const V=new Qe(0);let X=0,q=t.width,J=t.height,z=1,H=null,D=null;const W=new ct(0,0,q,J),Q=new ct(0,0,q,J);let me=!1;const De=new Wc;let Ie=!1,j=!1;const te=new ft,xe=new R,Fe=new ct,Te={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let Je=!1;function kt(){return L===null?z:1}let I=i;function _t(S,F){return t.getContext(S,F)}try{const S={alpha:!0,depth:s,stencil:r,antialias:a,premultipliedAlpha:l,preserveDrawingBuffer:c,powerPreference:u,failIfMajorPerformanceCaveat:h};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${Ic}`),t.addEventListener("webglcontextlost",he,!1),t.addEventListener("webglcontextrestored",ye,!1),t.addEventListener("webglcontextcreationerror",ie,!1),I===null){const F="webgl2";if(I=_t(F,S),I===null)throw _t(F)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(S){throw console.error("THREE.WebGLRenderer: "+S.message),S}let He,ke,Me,vt,we,Xe,Nt,wt,A,v,B,K,ne,Z,Re,ce,Ee,Ae,re,ge,Oe,Ce,fe,Ge;function U(){He=new py(I),He.init(),Ce=new ZS(I,He),ke=new ay(I,He,e,Ce),Me=new qS(I,He),ke.reversedDepthBuffer&&d&&Me.buffers.depth.setReversed(!0),vt=new _y(I),we=new US,Xe=new jS(I,He,Me,we,ke,Ce,vt),Nt=new ly(x),wt=new fy(x),A=new Mv(I),fe=new ry(I,A),v=new my(I,A,vt,fe),B=new xy(I,v,A,vt),re=new vy(I,ke,Xe),ce=new cy(we),K=new NS(x,Nt,wt,He,ke,fe,ce),ne=new nb(x,we),Z=new FS,Re=new VS(He),Ae=new sy(x,Nt,wt,Me,B,f,l),Ee=new $S(x,B,ke),Ge=new ib(I,vt,ke,Me),ge=new oy(I,He,vt),Oe=new gy(I,He,vt),vt.programs=K.programs,x.capabilities=ke,x.extensions=He,x.properties=we,x.renderLists=Z,x.shadowMap=Ee,x.state=Me,x.info=vt}U();const oe=new eb(x,I);this.xr=oe,this.getContext=function(){return I},this.getContextAttributes=function(){return I.getContextAttributes()},this.forceContextLoss=function(){const S=He.get("WEBGL_lose_context");S&&S.loseContext()},this.forceContextRestore=function(){const S=He.get("WEBGL_lose_context");S&&S.restoreContext()},this.getPixelRatio=function(){return z},this.setPixelRatio=function(S){S!==void 0&&(z=S,this.setSize(q,J,!1))},this.getSize=function(S){return S.set(q,J)},this.setSize=function(S,F,$=!0){if(oe.isPresenting){console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");return}q=S,J=F,t.width=Math.floor(S*z),t.height=Math.floor(F*z),$===!0&&(t.style.width=S+"px",t.style.height=F+"px"),this.setViewport(0,0,S,F)},this.getDrawingBufferSize=function(S){return S.set(q*z,J*z).floor()},this.setDrawingBufferSize=function(S,F,$){q=S,J=F,z=$,t.width=Math.floor(S*$),t.height=Math.floor(F*$),this.setViewport(0,0,S,F)},this.getCurrentViewport=function(S){return S.copy(N)},this.getViewport=function(S){return S.copy(W)},this.setViewport=function(S,F,$,Y){S.isVector4?W.set(S.x,S.y,S.z,S.w):W.set(S,F,$,Y),Me.viewport(N.copy(W).multiplyScalar(z).round())},this.getScissor=function(S){return S.copy(Q)},this.setScissor=function(S,F,$,Y){S.isVector4?Q.set(S.x,S.y,S.z,S.w):Q.set(S,F,$,Y),Me.scissor(O.copy(Q).multiplyScalar(z).round())},this.getScissorTest=function(){return me},this.setScissorTest=function(S){Me.setScissorTest(me=S)},this.setOpaqueSort=function(S){H=S},this.setTransparentSort=function(S){D=S},this.getClearColor=function(S){return S.copy(Ae.getClearColor())},this.setClearColor=function(){Ae.setClearColor(...arguments)},this.getClearAlpha=function(){return Ae.getClearAlpha()},this.setClearAlpha=function(){Ae.setClearAlpha(...arguments)},this.clear=function(S=!0,F=!0,$=!0){let Y=0;if(S){let k=!1;if(L!==null){const se=L.texture.format;k=se===zc||se===kc||se===Fc}if(k){const se=L.texture.type,pe=se===Rn||se===Ei||se===Bs||se===Hs||se===Uc||se===Oc,Se=Ae.getClearColor(),ve=Ae.getClearAlpha(),Ne=Se.r,ze=Se.g,Pe=Se.b;pe?(g[0]=Ne,g[1]=ze,g[2]=Pe,g[3]=ve,I.clearBufferuiv(I.COLOR,0,g)):(_[0]=Ne,_[1]=ze,_[2]=Pe,_[3]=ve,I.clearBufferiv(I.COLOR,0,_))}else Y|=I.COLOR_BUFFER_BIT}F&&(Y|=I.DEPTH_BUFFER_BIT),$&&(Y|=I.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),I.clear(Y)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",he,!1),t.removeEventListener("webglcontextrestored",ye,!1),t.removeEventListener("webglcontextcreationerror",ie,!1),Ae.dispose(),Z.dispose(),Re.dispose(),we.dispose(),Nt.dispose(),wt.dispose(),B.dispose(),fe.dispose(),Ge.dispose(),K.dispose(),oe.dispose(),oe.removeEventListener("sessionstart",yn),oe.removeEventListener("sessionend",el),li.stop()};function he(S){S.preventDefault(),console.log("THREE.WebGLRenderer: Context Lost."),P=!0}function ye(){console.log("THREE.WebGLRenderer: Context Restored."),P=!1;const S=vt.autoReset,F=Ee.enabled,$=Ee.autoUpdate,Y=Ee.needsUpdate,k=Ee.type;U(),vt.autoReset=S,Ee.enabled=F,Ee.autoUpdate=$,Ee.needsUpdate=Y,Ee.type=k}function ie(S){console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ",S.statusMessage)}function ee(S){const F=S.target;F.removeEventListener("dispose",ee),be(F)}function be(S){Be(S),we.remove(S)}function Be(S){const F=we.get(S).programs;F!==void 0&&(F.forEach(function($){K.releaseProgram($)}),S.isShaderMaterial&&K.releaseShaderCache(S))}this.renderBufferDirect=function(S,F,$,Y,k,se){F===null&&(F=Te);const pe=k.isMesh&&k.matrixWorld.determinant()<0,Se=Dd(S,F,$,Y,k);Me.setMaterial(Y,pe);let ve=$.index,Ne=1;if(Y.wireframe===!0){if(ve=v.getWireframeAttribute($),ve===void 0)return;Ne=2}const ze=$.drawRange,Pe=$.attributes.position;let Ke=ze.start*Ne,st=(ze.start+ze.count)*Ne;se!==null&&(Ke=Math.max(Ke,se.start*Ne),st=Math.min(st,(se.start+se.count)*Ne)),ve!==null?(Ke=Math.max(Ke,0),st=Math.min(st,ve.count)):Pe!=null&&(Ke=Math.max(Ke,0),st=Math.min(st,Pe.count));const Mt=st-Ke;if(Mt<0||Mt===1/0)return;fe.setup(k,Y,Se,$,ve);let pt,ut=ge;if(ve!==null&&(pt=A.get(ve),ut=Oe,ut.setIndex(pt)),k.isMesh)Y.wireframe===!0?(Me.setLineWidth(Y.wireframeLinewidth*kt()),ut.setMode(I.LINES)):ut.setMode(I.TRIANGLES);else if(k.isLine){let Le=Y.linewidth;Le===void 0&&(Le=1),Me.setLineWidth(Le*kt()),k.isLineSegments?ut.setMode(I.LINES):k.isLineLoop?ut.setMode(I.LINE_LOOP):ut.setMode(I.LINE_STRIP)}else k.isPoints?ut.setMode(I.POINTS):k.isSprite&&ut.setMode(I.TRIANGLES);if(k.isBatchedMesh)if(k._multiDrawInstances!==null)Xs("THREE.WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),ut.renderMultiDrawInstances(k._multiDrawStarts,k._multiDrawCounts,k._multiDrawCount,k._multiDrawInstances);else if(He.get("WEBGL_multi_draw"))ut.renderMultiDraw(k._multiDrawStarts,k._multiDrawCounts,k._multiDrawCount);else{const Le=k._multiDrawStarts,yt=k._multiDrawCounts,et=k._multiDrawCount,Kt=ve?A.get(ve).bytesPerElement:1,Pi=we.get(Y).currentProgram.getUniforms();for(let Qt=0;Qt<et;Qt++)Pi.setValue(I,"_gl_DrawID",Qt),ut.render(Le[Qt]/Kt,yt[Qt])}else if(k.isInstancedMesh)ut.renderInstances(Ke,Mt,k.count);else if($.isInstancedBufferGeometry){const Le=$._maxInstanceCount!==void 0?$._maxInstanceCount:1/0,yt=Math.min($.instanceCount,Le);ut.renderInstances(Ke,Mt,yt)}else ut.render(Ke,Mt)};function dt(S,F,$){S.transparent===!0&&S.side===cn&&S.forceSinglePass===!1?(S.side=Jt,S.needsUpdate=!0,sr(S,F,$),S.side=oi,S.needsUpdate=!0,sr(S,F,$),S.side=cn):sr(S,F,$)}this.compile=function(S,F,$=null){$===null&&($=S),p=Re.get($),p.init(F),M.push(p),$.traverseVisible(function(k){k.isLight&&k.layers.test(F.layers)&&(p.pushLight(k),k.castShadow&&p.pushShadow(k))}),S!==$&&S.traverseVisible(function(k){k.isLight&&k.layers.test(F.layers)&&(p.pushLight(k),k.castShadow&&p.pushShadow(k))}),p.setupLights();const Y=new Set;return S.traverse(function(k){if(!(k.isMesh||k.isPoints||k.isLine||k.isSprite))return;const se=k.material;if(se)if(Array.isArray(se))for(let pe=0;pe<se.length;pe++){const Se=se[pe];dt(Se,$,k),Y.add(Se)}else dt(se,$,k),Y.add(se)}),p=M.pop(),Y},this.compileAsync=function(S,F,$=null){const Y=this.compile(S,F,$);return new Promise(k=>{function se(){if(Y.forEach(function(pe){we.get(pe).currentProgram.isReady()&&Y.delete(pe)}),Y.size===0){k(S);return}setTimeout(se,10)}He.get("KHR_parallel_shader_compile")!==null?se():setTimeout(se,10)})};let nt=null;function Nn(S){nt&&nt(S)}function yn(){li.stop()}function el(){li.start()}const li=new Sd;li.setAnimationLoop(Nn),typeof self<"u"&&li.setContext(self),this.setAnimationLoop=function(S){nt=S,oe.setAnimationLoop(S),S===null?li.stop():li.start()},oe.addEventListener("sessionstart",yn),oe.addEventListener("sessionend",el),this.render=function(S,F){if(F!==void 0&&F.isCamera!==!0){console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(P===!0)return;if(S.matrixWorldAutoUpdate===!0&&S.updateMatrixWorld(),F.parent===null&&F.matrixWorldAutoUpdate===!0&&F.updateMatrixWorld(),oe.enabled===!0&&oe.isPresenting===!0&&(oe.cameraAutoUpdate===!0&&oe.updateCamera(F),F=oe.getCamera()),S.isScene===!0&&S.onBeforeRender(x,S,F,L),p=Re.get(S,M.length),p.init(F),M.push(p),te.multiplyMatrices(F.projectionMatrix,F.matrixWorldInverse),De.setFromProjectionMatrix(te,An,F.reversedDepth),j=this.localClippingEnabled,Ie=ce.init(this.clippingPlanes,j),m=Z.get(S,C.length),m.init(),C.push(m),oe.enabled===!0&&oe.isPresenting===!0){const se=x.xr.getDepthSensingMesh();se!==null&&Eo(se,F,-1/0,x.sortObjects)}Eo(S,F,0,x.sortObjects),m.finish(),x.sortObjects===!0&&m.sort(H,D),Je=oe.enabled===!1||oe.isPresenting===!1||oe.hasDepthSensing()===!1,Je&&Ae.addToRenderList(m,S),this.info.render.frame++,Ie===!0&&ce.beginShadows();const $=p.state.shadowsArray;Ee.render($,S,F),Ie===!0&&ce.endShadows(),this.info.autoReset===!0&&this.info.reset();const Y=m.opaque,k=m.transmissive;if(p.setupLights(),F.isArrayCamera){const se=F.cameras;if(k.length>0)for(let pe=0,Se=se.length;pe<Se;pe++){const ve=se[pe];nl(Y,k,S,ve)}Je&&Ae.render(S);for(let pe=0,Se=se.length;pe<Se;pe++){const ve=se[pe];tl(m,S,ve,ve.viewport)}}else k.length>0&&nl(Y,k,S,F),Je&&Ae.render(S),tl(m,S,F);L!==null&&T===0&&(Xe.updateMultisampleRenderTarget(L),Xe.updateRenderTargetMipmap(L)),S.isScene===!0&&S.onAfterRender(x,S,F),fe.resetDefaultState(),y=-1,b=null,M.pop(),M.length>0?(p=M[M.length-1],Ie===!0&&ce.setGlobalState(x.clippingPlanes,p.state.camera)):p=null,C.pop(),C.length>0?m=C[C.length-1]:m=null};function Eo(S,F,$,Y){if(S.visible===!1)return;if(S.layers.test(F.layers)){if(S.isGroup)$=S.renderOrder;else if(S.isLOD)S.autoUpdate===!0&&S.update(F);else if(S.isLight)p.pushLight(S),S.castShadow&&p.pushShadow(S);else if(S.isSprite){if(!S.frustumCulled||De.intersectsSprite(S)){Y&&Fe.setFromMatrixPosition(S.matrixWorld).applyMatrix4(te);const pe=B.update(S),Se=S.material;Se.visible&&m.push(S,pe,Se,$,Fe.z,null)}}else if((S.isMesh||S.isLine||S.isPoints)&&(!S.frustumCulled||De.intersectsObject(S))){const pe=B.update(S),Se=S.material;if(Y&&(S.boundingSphere!==void 0?(S.boundingSphere===null&&S.computeBoundingSphere(),Fe.copy(S.boundingSphere.center)):(pe.boundingSphere===null&&pe.computeBoundingSphere(),Fe.copy(pe.boundingSphere.center)),Fe.applyMatrix4(S.matrixWorld).applyMatrix4(te)),Array.isArray(Se)){const ve=pe.groups;for(let Ne=0,ze=ve.length;Ne<ze;Ne++){const Pe=ve[Ne],Ke=Se[Pe.materialIndex];Ke&&Ke.visible&&m.push(S,pe,Ke,$,Fe.z,Pe)}}else Se.visible&&m.push(S,pe,Se,$,Fe.z,null)}}const se=S.children;for(let pe=0,Se=se.length;pe<Se;pe++)Eo(se[pe],F,$,Y)}function tl(S,F,$,Y){const k=S.opaque,se=S.transmissive,pe=S.transparent;p.setupLightsView($),Ie===!0&&ce.setGlobalState(x.clippingPlanes,$),Y&&Me.viewport(N.copy(Y)),k.length>0&&ir(k,F,$),se.length>0&&ir(se,F,$),pe.length>0&&ir(pe,F,$),Me.buffers.depth.setTest(!0),Me.buffers.depth.setMask(!0),Me.buffers.color.setMask(!0),Me.setPolygonOffset(!1)}function nl(S,F,$,Y){if(($.isScene===!0?$.overrideMaterial:null)!==null)return;p.state.transmissionRenderTarget[Y.id]===void 0&&(p.state.transmissionRenderTarget[Y.id]=new Ai(1,1,{generateMipmaps:!0,type:He.has("EXT_color_buffer_half_float")||He.has("EXT_color_buffer_float")?Ks:Rn,minFilter:wi,samples:4,stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:tt.workingColorSpace}));const se=p.state.transmissionRenderTarget[Y.id],pe=Y.viewport||N;se.setSize(pe.z*x.transmissionResolutionScale,pe.w*x.transmissionResolutionScale);const Se=x.getRenderTarget(),ve=x.getActiveCubeFace(),Ne=x.getActiveMipmapLevel();x.setRenderTarget(se),x.getClearColor(V),X=x.getClearAlpha(),X<1&&x.setClearColor(16777215,.5),x.clear(),Je&&Ae.render($);const ze=x.toneMapping;x.toneMapping=ri;const Pe=Y.viewport;if(Y.viewport!==void 0&&(Y.viewport=void 0),p.setupLightsView(Y),Ie===!0&&ce.setGlobalState(x.clippingPlanes,Y),ir(S,$,Y),Xe.updateMultisampleRenderTarget(se),Xe.updateRenderTargetMipmap(se),He.has("WEBGL_multisampled_render_to_texture")===!1){let Ke=!1;for(let st=0,Mt=F.length;st<Mt;st++){const pt=F[st],ut=pt.object,Le=pt.geometry,yt=pt.material,et=pt.group;if(yt.side===cn&&ut.layers.test(Y.layers)){const Kt=yt.side;yt.side=Jt,yt.needsUpdate=!0,il(ut,$,Y,Le,yt,et),yt.side=Kt,yt.needsUpdate=!0,Ke=!0}}Ke===!0&&(Xe.updateMultisampleRenderTarget(se),Xe.updateRenderTargetMipmap(se))}x.setRenderTarget(Se,ve,Ne),x.setClearColor(V,X),Pe!==void 0&&(Y.viewport=Pe),x.toneMapping=ze}function ir(S,F,$){const Y=F.isScene===!0?F.overrideMaterial:null;for(let k=0,se=S.length;k<se;k++){const pe=S[k],Se=pe.object,ve=pe.geometry,Ne=pe.group;let ze=pe.material;ze.allowOverride===!0&&Y!==null&&(ze=Y),Se.layers.test($.layers)&&il(Se,F,$,ve,ze,Ne)}}function il(S,F,$,Y,k,se){S.onBeforeRender(x,F,$,Y,k,se),S.modelViewMatrix.multiplyMatrices($.matrixWorldInverse,S.matrixWorld),S.normalMatrix.getNormalMatrix(S.modelViewMatrix),k.onBeforeRender(x,F,$,Y,S,se),k.transparent===!0&&k.side===cn&&k.forceSinglePass===!1?(k.side=Jt,k.needsUpdate=!0,x.renderBufferDirect($,F,Y,k,S,se),k.side=oi,k.needsUpdate=!0,x.renderBufferDirect($,F,Y,k,S,se),k.side=cn):x.renderBufferDirect($,F,Y,k,S,se),S.onAfterRender(x,F,$,Y,k,se)}function sr(S,F,$){F.isScene!==!0&&(F=Te);const Y=we.get(S),k=p.state.lights,se=p.state.shadowsArray,pe=k.state.version,Se=K.getParameters(S,k.state,se,F,$),ve=K.getProgramCacheKey(Se);let Ne=Y.programs;Y.environment=S.isMeshStandardMaterial?F.environment:null,Y.fog=F.fog,Y.envMap=(S.isMeshStandardMaterial?wt:Nt).get(S.envMap||Y.environment),Y.envMapRotation=Y.environment!==null&&S.envMap===null?F.environmentRotation:S.envMapRotation,Ne===void 0&&(S.addEventListener("dispose",ee),Ne=new Map,Y.programs=Ne);let ze=Ne.get(ve);if(ze!==void 0){if(Y.currentProgram===ze&&Y.lightsStateVersion===pe)return rl(S,Se),ze}else Se.uniforms=K.getUniforms(S),S.onBeforeCompile(Se,x),ze=K.acquireProgram(Se,ve),Ne.set(ve,ze),Y.uniforms=Se.uniforms;const Pe=Y.uniforms;return(!S.isShaderMaterial&&!S.isRawShaderMaterial||S.clipping===!0)&&(Pe.clippingPlanes=ce.uniform),rl(S,Se),Y.needsLights=Nd(S),Y.lightsStateVersion=pe,Y.needsLights&&(Pe.ambientLightColor.value=k.state.ambient,Pe.lightProbe.value=k.state.probe,Pe.directionalLights.value=k.state.directional,Pe.directionalLightShadows.value=k.state.directionalShadow,Pe.spotLights.value=k.state.spot,Pe.spotLightShadows.value=k.state.spotShadow,Pe.rectAreaLights.value=k.state.rectArea,Pe.ltc_1.value=k.state.rectAreaLTC1,Pe.ltc_2.value=k.state.rectAreaLTC2,Pe.pointLights.value=k.state.point,Pe.pointLightShadows.value=k.state.pointShadow,Pe.hemisphereLights.value=k.state.hemi,Pe.directionalShadowMap.value=k.state.directionalShadowMap,Pe.directionalShadowMatrix.value=k.state.directionalShadowMatrix,Pe.spotShadowMap.value=k.state.spotShadowMap,Pe.spotLightMatrix.value=k.state.spotLightMatrix,Pe.spotLightMap.value=k.state.spotLightMap,Pe.pointShadowMap.value=k.state.pointShadowMap,Pe.pointShadowMatrix.value=k.state.pointShadowMatrix),Y.currentProgram=ze,Y.uniformsList=null,ze}function sl(S){if(S.uniformsList===null){const F=S.currentProgram.getUniforms();S.uniformsList=Qr.seqWithValue(F.seq,S.uniforms)}return S.uniformsList}function rl(S,F){const $=we.get(S);$.outputColorSpace=F.outputColorSpace,$.batching=F.batching,$.batchingColor=F.batchingColor,$.instancing=F.instancing,$.instancingColor=F.instancingColor,$.instancingMorph=F.instancingMorph,$.skinning=F.skinning,$.morphTargets=F.morphTargets,$.morphNormals=F.morphNormals,$.morphColors=F.morphColors,$.morphTargetsCount=F.morphTargetsCount,$.numClippingPlanes=F.numClippingPlanes,$.numIntersection=F.numClipIntersection,$.vertexAlphas=F.vertexAlphas,$.vertexTangents=F.vertexTangents,$.toneMapping=F.toneMapping}function Dd(S,F,$,Y,k){F.isScene!==!0&&(F=Te),Xe.resetTextureUnits();const se=F.fog,pe=Y.isMeshStandardMaterial?F.environment:null,Se=L===null?x.outputColorSpace:L.isXRRenderTarget===!0?L.texture.colorSpace:us,ve=(Y.isMeshStandardMaterial?wt:Nt).get(Y.envMap||pe),Ne=Y.vertexColors===!0&&!!$.attributes.color&&$.attributes.color.itemSize===4,ze=!!$.attributes.tangent&&(!!Y.normalMap||Y.anisotropy>0),Pe=!!$.morphAttributes.position,Ke=!!$.morphAttributes.normal,st=!!$.morphAttributes.color;let Mt=ri;Y.toneMapped&&(L===null||L.isXRRenderTarget===!0)&&(Mt=x.toneMapping);const pt=$.morphAttributes.position||$.morphAttributes.normal||$.morphAttributes.color,ut=pt!==void 0?pt.length:0,Le=we.get(Y),yt=p.state.lights;if(Ie===!0&&(j===!0||S!==b)){const Wt=S===b&&Y.id===y;ce.setState(Y,S,Wt)}let et=!1;Y.version===Le.__version?(Le.needsLights&&Le.lightsStateVersion!==yt.state.version||Le.outputColorSpace!==Se||k.isBatchedMesh&&Le.batching===!1||!k.isBatchedMesh&&Le.batching===!0||k.isBatchedMesh&&Le.batchingColor===!0&&k.colorTexture===null||k.isBatchedMesh&&Le.batchingColor===!1&&k.colorTexture!==null||k.isInstancedMesh&&Le.instancing===!1||!k.isInstancedMesh&&Le.instancing===!0||k.isSkinnedMesh&&Le.skinning===!1||!k.isSkinnedMesh&&Le.skinning===!0||k.isInstancedMesh&&Le.instancingColor===!0&&k.instanceColor===null||k.isInstancedMesh&&Le.instancingColor===!1&&k.instanceColor!==null||k.isInstancedMesh&&Le.instancingMorph===!0&&k.morphTexture===null||k.isInstancedMesh&&Le.instancingMorph===!1&&k.morphTexture!==null||Le.envMap!==ve||Y.fog===!0&&Le.fog!==se||Le.numClippingPlanes!==void 0&&(Le.numClippingPlanes!==ce.numPlanes||Le.numIntersection!==ce.numIntersection)||Le.vertexAlphas!==Ne||Le.vertexTangents!==ze||Le.morphTargets!==Pe||Le.morphNormals!==Ke||Le.morphColors!==st||Le.toneMapping!==Mt||Le.morphTargetsCount!==ut)&&(et=!0):(et=!0,Le.__version=Y.version);let Kt=Le.currentProgram;et===!0&&(Kt=sr(Y,F,k));let Pi=!1,Qt=!1,_s=!1;const St=Kt.getUniforms(),rn=Le.uniforms;if(Me.useProgram(Kt.program)&&(Pi=!0,Qt=!0,_s=!0),Y.id!==y&&(y=Y.id,Qt=!0),Pi||b!==S){Me.buffers.depth.getReversed()&&S.reversedDepth!==!0&&(S._reversedDepth=!0,S.updateProjectionMatrix()),St.setValue(I,"projectionMatrix",S.projectionMatrix),St.setValue(I,"viewMatrix",S.matrixWorldInverse);const jt=St.map.cameraPosition;jt!==void 0&&jt.setValue(I,xe.setFromMatrixPosition(S.matrixWorld)),ke.logarithmicDepthBuffer&&St.setValue(I,"logDepthBufFC",2/(Math.log(S.far+1)/Math.LN2)),(Y.isMeshPhongMaterial||Y.isMeshToonMaterial||Y.isMeshLambertMaterial||Y.isMeshBasicMaterial||Y.isMeshStandardMaterial||Y.isShaderMaterial)&&St.setValue(I,"isOrthographic",S.isOrthographicCamera===!0),b!==S&&(b=S,Qt=!0,_s=!0)}if(k.isSkinnedMesh){St.setOptional(I,k,"bindMatrix"),St.setOptional(I,k,"bindMatrixInverse");const Wt=k.skeleton;Wt&&(Wt.boneTexture===null&&Wt.computeBoneTexture(),St.setValue(I,"boneTexture",Wt.boneTexture,Xe))}k.isBatchedMesh&&(St.setOptional(I,k,"batchingTexture"),St.setValue(I,"batchingTexture",k._matricesTexture,Xe),St.setOptional(I,k,"batchingIdTexture"),St.setValue(I,"batchingIdTexture",k._indirectTexture,Xe),St.setOptional(I,k,"batchingColorTexture"),k._colorsTexture!==null&&St.setValue(I,"batchingColorTexture",k._colorsTexture,Xe));const on=$.morphAttributes;if((on.position!==void 0||on.normal!==void 0||on.color!==void 0)&&re.update(k,$,Kt),(Qt||Le.receiveShadow!==k.receiveShadow)&&(Le.receiveShadow=k.receiveShadow,St.setValue(I,"receiveShadow",k.receiveShadow)),Y.isMeshGouraudMaterial&&Y.envMap!==null&&(rn.envMap.value=ve,rn.flipEnvMap.value=ve.isCubeTexture&&ve.isRenderTargetTexture===!1?-1:1),Y.isMeshStandardMaterial&&Y.envMap===null&&F.environment!==null&&(rn.envMapIntensity.value=F.environmentIntensity),Qt&&(St.setValue(I,"toneMappingExposure",x.toneMappingExposure),Le.needsLights&&Id(rn,_s),se&&Y.fog===!0&&ne.refreshFogUniforms(rn,se),ne.refreshMaterialUniforms(rn,Y,z,J,p.state.transmissionRenderTarget[S.id]),Qr.upload(I,sl(Le),rn,Xe)),Y.isShaderMaterial&&Y.uniformsNeedUpdate===!0&&(Qr.upload(I,sl(Le),rn,Xe),Y.uniformsNeedUpdate=!1),Y.isSpriteMaterial&&St.setValue(I,"center",k.center),St.setValue(I,"modelViewMatrix",k.modelViewMatrix),St.setValue(I,"normalMatrix",k.normalMatrix),St.setValue(I,"modelMatrix",k.matrixWorld),Y.isShaderMaterial||Y.isRawShaderMaterial){const Wt=Y.uniformsGroups;for(let jt=0,To=Wt.length;jt<To;jt++){const ui=Wt[jt];Ge.update(ui,Kt),Ge.bind(ui,Kt)}}return Kt}function Id(S,F){S.ambientLightColor.needsUpdate=F,S.lightProbe.needsUpdate=F,S.directionalLights.needsUpdate=F,S.directionalLightShadows.needsUpdate=F,S.pointLights.needsUpdate=F,S.pointLightShadows.needsUpdate=F,S.spotLights.needsUpdate=F,S.spotLightShadows.needsUpdate=F,S.rectAreaLights.needsUpdate=F,S.hemisphereLights.needsUpdate=F}function Nd(S){return S.isMeshLambertMaterial||S.isMeshToonMaterial||S.isMeshPhongMaterial||S.isMeshStandardMaterial||S.isShadowMaterial||S.isShaderMaterial&&S.lights===!0}this.getActiveCubeFace=function(){return w},this.getActiveMipmapLevel=function(){return T},this.getRenderTarget=function(){return L},this.setRenderTargetTextures=function(S,F,$){const Y=we.get(S);Y.__autoAllocateDepthBuffer=S.resolveDepthBuffer===!1,Y.__autoAllocateDepthBuffer===!1&&(Y.__useRenderToTexture=!1),we.get(S.texture).__webglTexture=F,we.get(S.depthTexture).__webglTexture=Y.__autoAllocateDepthBuffer?void 0:$,Y.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(S,F){const $=we.get(S);$.__webglFramebuffer=F,$.__useDefaultFramebuffer=F===void 0};const Ud=I.createFramebuffer();this.setRenderTarget=function(S,F=0,$=0){L=S,w=F,T=$;let Y=!0,k=null,se=!1,pe=!1;if(S){const ve=we.get(S);if(ve.__useDefaultFramebuffer!==void 0)Me.bindFramebuffer(I.FRAMEBUFFER,null),Y=!1;else if(ve.__webglFramebuffer===void 0)Xe.setupRenderTarget(S);else if(ve.__hasExternalTextures)Xe.rebindTextures(S,we.get(S.texture).__webglTexture,we.get(S.depthTexture).__webglTexture);else if(S.depthBuffer){const Pe=S.depthTexture;if(ve.__boundDepthTexture!==Pe){if(Pe!==null&&we.has(Pe)&&(S.width!==Pe.image.width||S.height!==Pe.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");Xe.setupDepthRenderbuffer(S)}}const Ne=S.texture;(Ne.isData3DTexture||Ne.isDataArrayTexture||Ne.isCompressedArrayTexture)&&(pe=!0);const ze=we.get(S).__webglFramebuffer;S.isWebGLCubeRenderTarget?(Array.isArray(ze[F])?k=ze[F][$]:k=ze[F],se=!0):S.samples>0&&Xe.useMultisampledRTT(S)===!1?k=we.get(S).__webglMultisampledFramebuffer:Array.isArray(ze)?k=ze[$]:k=ze,N.copy(S.viewport),O.copy(S.scissor),G=S.scissorTest}else N.copy(W).multiplyScalar(z).floor(),O.copy(Q).multiplyScalar(z).floor(),G=me;if($!==0&&(k=Ud),Me.bindFramebuffer(I.FRAMEBUFFER,k)&&Y&&Me.drawBuffers(S,k),Me.viewport(N),Me.scissor(O),Me.setScissorTest(G),se){const ve=we.get(S.texture);I.framebufferTexture2D(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_CUBE_MAP_POSITIVE_X+F,ve.__webglTexture,$)}else if(pe){const ve=F;for(let Ne=0;Ne<S.textures.length;Ne++){const ze=we.get(S.textures[Ne]);I.framebufferTextureLayer(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0+Ne,ze.__webglTexture,$,ve)}}else if(S!==null&&$!==0){const ve=we.get(S.texture);I.framebufferTexture2D(I.FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,ve.__webglTexture,$)}y=-1},this.readRenderTargetPixels=function(S,F,$,Y,k,se,pe,Se=0){if(!(S&&S.isWebGLRenderTarget)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ve=we.get(S).__webglFramebuffer;if(S.isWebGLCubeRenderTarget&&pe!==void 0&&(ve=ve[pe]),ve){Me.bindFramebuffer(I.FRAMEBUFFER,ve);try{const Ne=S.textures[Se],ze=Ne.format,Pe=Ne.type;if(!ke.textureFormatReadable(ze)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!ke.textureTypeReadable(Pe)){console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}F>=0&&F<=S.width-Y&&$>=0&&$<=S.height-k&&(S.textures.length>1&&I.readBuffer(I.COLOR_ATTACHMENT0+Se),I.readPixels(F,$,Y,k,Ce.convert(ze),Ce.convert(Pe),se))}finally{const Ne=L!==null?we.get(L).__webglFramebuffer:null;Me.bindFramebuffer(I.FRAMEBUFFER,Ne)}}},this.readRenderTargetPixelsAsync=async function(S,F,$,Y,k,se,pe,Se=0){if(!(S&&S.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ve=we.get(S).__webglFramebuffer;if(S.isWebGLCubeRenderTarget&&pe!==void 0&&(ve=ve[pe]),ve)if(F>=0&&F<=S.width-Y&&$>=0&&$<=S.height-k){Me.bindFramebuffer(I.FRAMEBUFFER,ve);const Ne=S.textures[Se],ze=Ne.format,Pe=Ne.type;if(!ke.textureFormatReadable(ze))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!ke.textureTypeReadable(Pe))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const Ke=I.createBuffer();I.bindBuffer(I.PIXEL_PACK_BUFFER,Ke),I.bufferData(I.PIXEL_PACK_BUFFER,se.byteLength,I.STREAM_READ),S.textures.length>1&&I.readBuffer(I.COLOR_ATTACHMENT0+Se),I.readPixels(F,$,Y,k,Ce.convert(ze),Ce.convert(Pe),0);const st=L!==null?we.get(L).__webglFramebuffer:null;Me.bindFramebuffer(I.FRAMEBUFFER,st);const Mt=I.fenceSync(I.SYNC_GPU_COMMANDS_COMPLETE,0);return I.flush(),await e_(I,Mt,4),I.bindBuffer(I.PIXEL_PACK_BUFFER,Ke),I.getBufferSubData(I.PIXEL_PACK_BUFFER,0,se),I.deleteBuffer(Ke),I.deleteSync(Mt),se}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(S,F=null,$=0){const Y=Math.pow(2,-$),k=Math.floor(S.image.width*Y),se=Math.floor(S.image.height*Y),pe=F!==null?F.x:0,Se=F!==null?F.y:0;Xe.setTexture2D(S,0),I.copyTexSubImage2D(I.TEXTURE_2D,$,0,0,pe,Se,k,se),Me.unbindTexture()};const Od=I.createFramebuffer(),Fd=I.createFramebuffer();this.copyTextureToTexture=function(S,F,$=null,Y=null,k=0,se=null){se===null&&(k!==0?(Xs("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."),se=k,k=0):se=0);let pe,Se,ve,Ne,ze,Pe,Ke,st,Mt;const pt=S.isCompressedTexture?S.mipmaps[se]:S.image;if($!==null)pe=$.max.x-$.min.x,Se=$.max.y-$.min.y,ve=$.isBox3?$.max.z-$.min.z:1,Ne=$.min.x,ze=$.min.y,Pe=$.isBox3?$.min.z:0;else{const on=Math.pow(2,-k);pe=Math.floor(pt.width*on),Se=Math.floor(pt.height*on),S.isDataArrayTexture?ve=pt.depth:S.isData3DTexture?ve=Math.floor(pt.depth*on):ve=1,Ne=0,ze=0,Pe=0}Y!==null?(Ke=Y.x,st=Y.y,Mt=Y.z):(Ke=0,st=0,Mt=0);const ut=Ce.convert(F.format),Le=Ce.convert(F.type);let yt;F.isData3DTexture?(Xe.setTexture3D(F,0),yt=I.TEXTURE_3D):F.isDataArrayTexture||F.isCompressedArrayTexture?(Xe.setTexture2DArray(F,0),yt=I.TEXTURE_2D_ARRAY):(Xe.setTexture2D(F,0),yt=I.TEXTURE_2D),I.pixelStorei(I.UNPACK_FLIP_Y_WEBGL,F.flipY),I.pixelStorei(I.UNPACK_PREMULTIPLY_ALPHA_WEBGL,F.premultiplyAlpha),I.pixelStorei(I.UNPACK_ALIGNMENT,F.unpackAlignment);const et=I.getParameter(I.UNPACK_ROW_LENGTH),Kt=I.getParameter(I.UNPACK_IMAGE_HEIGHT),Pi=I.getParameter(I.UNPACK_SKIP_PIXELS),Qt=I.getParameter(I.UNPACK_SKIP_ROWS),_s=I.getParameter(I.UNPACK_SKIP_IMAGES);I.pixelStorei(I.UNPACK_ROW_LENGTH,pt.width),I.pixelStorei(I.UNPACK_IMAGE_HEIGHT,pt.height),I.pixelStorei(I.UNPACK_SKIP_PIXELS,Ne),I.pixelStorei(I.UNPACK_SKIP_ROWS,ze),I.pixelStorei(I.UNPACK_SKIP_IMAGES,Pe);const St=S.isDataArrayTexture||S.isData3DTexture,rn=F.isDataArrayTexture||F.isData3DTexture;if(S.isDepthTexture){const on=we.get(S),Wt=we.get(F),jt=we.get(on.__renderTarget),To=we.get(Wt.__renderTarget);Me.bindFramebuffer(I.READ_FRAMEBUFFER,jt.__webglFramebuffer),Me.bindFramebuffer(I.DRAW_FRAMEBUFFER,To.__webglFramebuffer);for(let ui=0;ui<ve;ui++)St&&(I.framebufferTextureLayer(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,we.get(S).__webglTexture,k,Pe+ui),I.framebufferTextureLayer(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,we.get(F).__webglTexture,se,Mt+ui)),I.blitFramebuffer(Ne,ze,pe,Se,Ke,st,pe,Se,I.DEPTH_BUFFER_BIT,I.NEAREST);Me.bindFramebuffer(I.READ_FRAMEBUFFER,null),Me.bindFramebuffer(I.DRAW_FRAMEBUFFER,null)}else if(k!==0||S.isRenderTargetTexture||we.has(S)){const on=we.get(S),Wt=we.get(F);Me.bindFramebuffer(I.READ_FRAMEBUFFER,Od),Me.bindFramebuffer(I.DRAW_FRAMEBUFFER,Fd);for(let jt=0;jt<ve;jt++)St?I.framebufferTextureLayer(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,on.__webglTexture,k,Pe+jt):I.framebufferTexture2D(I.READ_FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,on.__webglTexture,k),rn?I.framebufferTextureLayer(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,Wt.__webglTexture,se,Mt+jt):I.framebufferTexture2D(I.DRAW_FRAMEBUFFER,I.COLOR_ATTACHMENT0,I.TEXTURE_2D,Wt.__webglTexture,se),k!==0?I.blitFramebuffer(Ne,ze,pe,Se,Ke,st,pe,Se,I.COLOR_BUFFER_BIT,I.NEAREST):rn?I.copyTexSubImage3D(yt,se,Ke,st,Mt+jt,Ne,ze,pe,Se):I.copyTexSubImage2D(yt,se,Ke,st,Ne,ze,pe,Se);Me.bindFramebuffer(I.READ_FRAMEBUFFER,null),Me.bindFramebuffer(I.DRAW_FRAMEBUFFER,null)}else rn?S.isDataTexture||S.isData3DTexture?I.texSubImage3D(yt,se,Ke,st,Mt,pe,Se,ve,ut,Le,pt.data):F.isCompressedArrayTexture?I.compressedTexSubImage3D(yt,se,Ke,st,Mt,pe,Se,ve,ut,pt.data):I.texSubImage3D(yt,se,Ke,st,Mt,pe,Se,ve,ut,Le,pt):S.isDataTexture?I.texSubImage2D(I.TEXTURE_2D,se,Ke,st,pe,Se,ut,Le,pt.data):S.isCompressedTexture?I.compressedTexSubImage2D(I.TEXTURE_2D,se,Ke,st,pt.width,pt.height,ut,pt.data):I.texSubImage2D(I.TEXTURE_2D,se,Ke,st,pe,Se,ut,Le,pt);I.pixelStorei(I.UNPACK_ROW_LENGTH,et),I.pixelStorei(I.UNPACK_IMAGE_HEIGHT,Kt),I.pixelStorei(I.UNPACK_SKIP_PIXELS,Pi),I.pixelStorei(I.UNPACK_SKIP_ROWS,Qt),I.pixelStorei(I.UNPACK_SKIP_IMAGES,_s),se===0&&F.generateMipmaps&&I.generateMipmap(yt),Me.unbindTexture()},this.initRenderTarget=function(S){we.get(S).__webglFramebuffer===void 0&&Xe.setupRenderTarget(S)},this.initTexture=function(S){S.isCubeTexture?Xe.setTextureCube(S,0):S.isData3DTexture?Xe.setTexture3D(S,0):S.isDataArrayTexture||S.isCompressedArrayTexture?Xe.setTexture2DArray(S,0):Xe.setTexture2D(S,0),Me.unbindTexture()},this.resetState=function(){w=0,T=0,L=null,Me.reset(),fe.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return An}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=tt._getDrawingBufferColorSpace(e),t.unpackColorSpace=tt._getUnpackColorSpace()}}const th=1710879,rb=2040103,ob=3817030,ab=2370101,cb=4608100,lb=2434861,ub=3290428,hb=14832981,db=4054148,fb=15132908,ks=16733525,pb=5087231,mb=.35,gb=9065813,Td=16733525;function va(n,e,t=6){const i=document.createElement("canvas"),s=i.getContext("2d"),r=48,o=`600 ${r}px system-ui, sans-serif`;s.font=o;const a=s.measureText(n),l=10;i.width=Math.ceil(a.width+l*2),i.height=Math.ceil(r*1.4),s.font=o,s.fillStyle=e,s.textAlign="center",s.textBaseline="middle",s.fillText(n,i.width/2,i.height/2);const c=new tr(i);c.colorSpace=Ot;const u=new Gc({map:c,transparent:!0,depthWrite:!1}),h=i.width/i.height,d=new Vc(u);return d.scale.set(t*h,t,1),d}const _b=10,vb=50;function xb(n){return{x:new R(n.x_axis_right?-1:1,0,0),y:new R(0,0,n.y_axis_down?1:-1)}}function Ji(n,e=1){return new Pn({color:n,transparent:e<1,opacity:e})}function xa(n,e,t,i,s,r){const o=[Pt({x:n,y:e},s),Pt({x:t,y:e},s),Pt({x:t,y:i},s),Pt({x:n,y:i},s)].map(l=>new R(l.x,l.y,l.z)),a=new it().setFromPoints(o);return new cd(a,r)}function nh(n,e,t,i,s,r,o){const a=t-n,l=i-e,c=new ms(a,l),u=new xt(c,new qt({color:r,transparent:o<1,opacity:o,depthWrite:o>=1}));return u.rotation.x=-Math.PI/2,u.position.set((n+t)/2,s,-(e+i)/2),u}function ih(n,e,t,i){const s=[];for(let r=0;r<=n+1e-6;r+=t)s.push(new R(r,i,0),new R(r,i,-e));for(let r=0;r<=e+1e-6;r+=t)s.push(new R(0,i,-r),new R(n,i,-r));return s}const Ad=.1;function yb(n,e=Ad){const t=[];for(let i=0;i+1<n.length;i+=2){const s=Pt({x:n[i],y:n[i+1]},e);t.push(new R(s.x,s.y,s.z))}return t}function Sb(n,e=Ad){if(n.length<6)return null;const t=new Mo;t.moveTo(n[0],n[1]);for(let s=2;s+1<n.length;s+=2)t.lineTo(n[s],n[s+1]);const i=new xt(new nr(t),new qt({color:Td,transparent:!0,opacity:.12,side:cn,depthWrite:!1}));return i.rotation.x=-Math.PI/2,i.position.y=e-.01,i}class bb{group=new Ft;gridGroup=new Ft;nogoGroup=null;nogoVisible=!0;density=null;vp=null;constructor(){this.group.add(this.gridGroup)}build(e){this.vp=e,this.nogoGroup=null,this.disposeChildren(this.group),this.gridGroup=new Ft,this.group.add(this.gridGroup);const t=e.extents.width_mm,i=e.extents.height_mm;this.group.add(nh(0,0,t,i,-.02,rb,1)),this.group.add(xa(0,0,t,i,.02,Ji(ob)));const s=e.workarea;if(this.group.add(nh(s.x_min,s.y_min,s.x_max,s.y_max,.06,ab,.6)),this.group.add(xa(s.x_min,s.y_min,s.x_max,s.y_max,.08,Ji(cb))),e.soft_limits_panel){const r=e.soft_limits_panel,o=new xd({color:gb,dashSize:4,gapSize:4,transparent:!0,opacity:.8}),a=xa(r.x,r.y,r.x+r.w,r.y+r.h,.09,o);a.computeLineDistances(),this.group.add(a)}this.buildAxisArrows(e),this.buildWorkOriginMarker(e.work_origin_panel),this.buildNogoZones(e.nogo_zones),this.setDensity(this.density??"fine")}setNogoZonesVisible(e){this.nogoVisible=e,this.nogoGroup&&(this.nogoGroup.visible=e)}buildNogoZones(e){if(!e||e.length===0)return;const t=new Ft;for(const i of e){if(i.length<6)continue;const s=Sb(i);s&&t.add(s),t.add(new cd(new it().setFromPoints(yb(i)),Ji(Td,.85)))}t.visible=this.nogoVisible,this.nogoGroup=t,this.group.add(t)}setDensity(e){if(e===this.density&&this.gridGroup.children.length>0)return;this.density=e,this.disposeChildren(this.gridGroup);const t=this.vp;if(!t)return;const i=t.extents.width_mm,s=t.extents.height_mm,r=ih(i,s,vb,.04),o=new it().setFromPoints(r);if(this.gridGroup.add(new ho(o,Ji(ub))),e==="fine"){const a=ih(i,s,_b,.03),l=new it().setFromPoints(a);this.gridGroup.add(new ho(l,Ji(lb)))}}buildAxisArrows(e){const t=e.workarea_origin_panel,i=Math.abs(e.workarea.x_max-e.workarea.x_min),s=Math.abs(e.workarea.y_max-e.workarea.y_min),r=Math.max(15,Math.min(i,s)*.25),o=r*.22,a=o*.5,l=xb(e),c=new Ru(l.x,new R(t.x,.12,-t.y),r,hb,o,a);this.group.add(c);const u=va("X","#e25555",r*.28),h=Pt({x:t.x+l.x.x*r*1.25,y:t.y},.12);u.position.set(h.x,h.y+r*.1,h.z),this.group.add(u);const d=new Ru(l.y,new R(t.x,.12,-t.y),r,db,o,a);this.group.add(d);const f=va("Y","#3ddc84",r*.28),g=Pt({x:t.x,y:t.y-l.y.z*r*1.25},.12);f.position.set(g.x,g.y+r*.1,g.z),this.group.add(f)}buildWorkOriginMarker(e){const t=new Ft,i=5,s=Ji(fb),r=(c,u)=>{const h=[Pt(c,.14),Pt(u,.14)].map(d=>new R(d.x,d.y,d.z));return new Ln(new it().setFromPoints(h),s)};t.add(r({x:e.x-i,y:e.y},{x:e.x+i,y:e.y})),t.add(r({x:e.x,y:e.y-i},{x:e.x,y:e.y+i}));const o=[];for(let c=0;c<=24;c++){const u=c/24*Math.PI*2;o.push(new R(e.x+Math.cos(u)*i,.14,-(e.y+Math.sin(u)*i)))}t.add(new Ln(new it().setFromPoints(o),s));const a=va("W","#e6e8ec",5),l=Pt({x:e.x+i+4,y:e.y+i+2},.14);a.position.set(l.x,l.y+3,l.z),t.add(a),this.group.add(t)}disposeChildren(e){const t=[...e.children];for(const i of t)e.remove(i),i.traverse(s=>{const r=s;r.geometry&&r.geometry.dispose();const o=r.material;o&&o.dispose()})}}class Mb{group=new Ft;cone;dot;light;material;constructor(){this.material=new qt({color:ks,transparent:!0,opacity:.55,depthWrite:!1});const e=new bo(1.1,9,16,1,!0);this.cone=new xt(e,this.material),this.cone.rotation.x=Math.PI,this.group.add(this.cone),this.dot=new xt(new $s(1.7,24),new qt({color:ks,transparent:!0,opacity:.95,depthWrite:!1})),this.dot.rotation.x=-Math.PI/2,this.group.add(this.dot),this.light=new pv(ks,40,60,2),this.group.add(this.light),this.group.visible=!1,this.group.renderOrder=5}setPosition(e){if(!e){this.group.visible=!1;return}const t=Pt(e,.25);this.group.visible=!0,this.dot.position.set(t.x,t.y,t.z),this.cone.position.set(t.x,t.y+4.5,t.z),this.light.position.set(t.x,t.y+2,t.z)}tick(e){if(!this.group.visible)return;const t=.45+.2*Math.sin(e*6);this.material.opacity=t,this.dot.material.opacity=.7+.3*t}}const sh={type:"change"},Jc={type:"start"},Cd={type:"end"},Gr=new yo,rh=new Gn,wb=Math.cos(70*Kg.DEG2RAD),Ct=new R,Zt=2*Math.PI,at={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},ya=1e-6;class Eb extends Sv{constructor(e,t=null){super(e,t),this.state=at.NONE,this.target=new R,this.cursor=new R,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:ts.ROTATE,MIDDLE:ts.DOLLY,RIGHT:ts.PAN},this.touches={ONE:Qi.ROTATE,TWO:Qi.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this._lastPosition=new R,this._lastQuaternion=new Ti,this._lastTargetPosition=new R,this._quat=new Ti().setFromUnitVectors(e.up,new R(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new Au,this._sphericalDelta=new Au,this._scale=1,this._panOffset=new R,this._rotateStart=new ue,this._rotateEnd=new ue,this._rotateDelta=new ue,this._panStart=new ue,this._panEnd=new ue,this._panDelta=new ue,this._dollyStart=new ue,this._dollyEnd=new ue,this._dollyDelta=new ue,this._dollyDirection=new R,this._mouse=new ue,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=Ab.bind(this),this._onPointerDown=Tb.bind(this),this._onPointerUp=Cb.bind(this),this._onContextMenu=Ub.bind(this),this._onMouseWheel=Lb.bind(this),this._onKeyDown=Db.bind(this),this._onTouchStart=Ib.bind(this),this._onTouchMove=Nb.bind(this),this._onMouseDown=Rb.bind(this),this._onMouseMove=Pb.bind(this),this._interceptControlDown=Ob.bind(this),this._interceptControlUp=Fb.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(sh),this.update(),this.state=at.NONE}update(e=null){const t=this.object.position;Ct.copy(t).sub(this.target),Ct.applyQuaternion(this._quat),this._spherical.setFromVector3(Ct),this.autoRotate&&this.state===at.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let i=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(i)&&isFinite(s)&&(i<-Math.PI?i+=Zt:i>Math.PI&&(i-=Zt),s<-Math.PI?s+=Zt:s>Math.PI&&(s-=Zt),i<=s?this._spherical.theta=Math.max(i,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(i+s)/2?Math.max(i,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{const o=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=o!=this._spherical.radius}if(Ct.setFromSpherical(this._spherical),Ct.applyQuaternion(this._quatInverse),t.copy(this.target).add(Ct),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let o=null;if(this.object.isPerspectiveCamera){const a=Ct.length();o=this._clampDistance(a*this._scale);const l=a-o;this.object.position.addScaledVector(this._dollyDirection,l),this.object.updateMatrixWorld(),r=!!l}else if(this.object.isOrthographicCamera){const a=new R(this._mouse.x,this._mouse.y,0);a.unproject(this.object);const l=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=l!==this.object.zoom;const c=new R(this._mouse.x,this._mouse.y,0);c.unproject(this.object),this.object.position.sub(c).add(a),this.object.updateMatrixWorld(),o=Ct.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;o!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(o).add(this.object.position):(Gr.origin.copy(this.object.position),Gr.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Gr.direction))<wb?this.object.lookAt(this.target):(rh.setFromNormalAndCoplanarPoint(this.object.up,this.target),Gr.intersectPlane(rh,this.target))))}else if(this.object.isOrthographicCamera){const o=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),o!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>ya||8*(1-this._lastQuaternion.dot(this.object.quaternion))>ya||this._lastTargetPosition.distanceToSquared(this.target)>ya?(this.dispatchEvent(sh),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?Zt/60*this.autoRotateSpeed*e:Zt/60/60*this.autoRotateSpeed}_getZoomScale(e){const t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){Ct.setFromMatrixColumn(t,0),Ct.multiplyScalar(-e),this._panOffset.add(Ct)}_panUp(e,t){this.screenSpacePanning===!0?Ct.setFromMatrixColumn(t,1):(Ct.setFromMatrixColumn(t,0),Ct.crossVectors(this.object.up,Ct)),Ct.multiplyScalar(e),this._panOffset.add(Ct)}_pan(e,t){const i=this.domElement;if(this.object.isPerspectiveCamera){const s=this.object.position;Ct.copy(s).sub(this.target);let r=Ct.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*r/i.clientHeight,this.object.matrix),this._panUp(2*t*r/i.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/i.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/i.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;const i=this.domElement.getBoundingClientRect(),s=e-i.left,r=t-i.top,o=i.width,a=i.height;this._mouse.x=s/o*2-1,this._mouse.y=-(r/a)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(Zt*this._rotateDelta.x/t.clientHeight),this._rotateUp(Zt*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(Zt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-Zt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(Zt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-Zt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._rotateStart.set(i,s)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panStart.set(i,s)}}_handleTouchStartDolly(e){const t=this._getSecondPointerPosition(e),i=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(i*i+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{const i=this._getSecondPointerPosition(e),s=.5*(e.pageX+i.x),r=.5*(e.pageY+i.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(Zt*this._rotateDelta.x/t.clientHeight),this._rotateUp(Zt*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),i=.5*(e.pageX+t.x),s=.5*(e.pageY+t.y);this._panEnd.set(i,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){const t=this._getSecondPointerPosition(e),i=e.pageX-t.x,s=e.pageY-t.y,r=Math.sqrt(i*i+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);const o=(e.pageX+t.x)*.5,a=(e.pageY+t.y)*.5;this._updateZoomParameters(o,a)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new ue,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){const t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){const t=e.deltaMode,i={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:i.deltaY*=16;break;case 2:i.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(i.deltaY*=10),i}}function Tb(n){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(n.pointerId),this.domElement.addEventListener("pointermove",this._onPointerMove),this.domElement.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(n)&&(this._addPointer(n),n.pointerType==="touch"?this._onTouchStart(n):this._onMouseDown(n)))}function Ab(n){this.enabled!==!1&&(n.pointerType==="touch"?this._onTouchMove(n):this._onMouseMove(n))}function Cb(n){switch(this._removePointer(n),this._pointers.length){case 0:this.domElement.releasePointerCapture(n.pointerId),this.domElement.removeEventListener("pointermove",this._onPointerMove),this.domElement.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Cd),this.state=at.NONE;break;case 1:const e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function Rb(n){let e;switch(n.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case ts.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(n),this.state=at.DOLLY;break;case ts.ROTATE:if(n.ctrlKey||n.metaKey||n.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(n),this.state=at.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(n),this.state=at.ROTATE}break;case ts.PAN:if(n.ctrlKey||n.metaKey||n.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(n),this.state=at.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(n),this.state=at.PAN}break;default:this.state=at.NONE}this.state!==at.NONE&&this.dispatchEvent(Jc)}function Pb(n){switch(this.state){case at.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(n);break;case at.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(n);break;case at.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(n);break}}function Lb(n){this.enabled===!1||this.enableZoom===!1||this.state!==at.NONE||(n.preventDefault(),this.dispatchEvent(Jc),this._handleMouseWheel(this._customWheelEvent(n)),this.dispatchEvent(Cd))}function Db(n){this.enabled!==!1&&this._handleKeyDown(n)}function Ib(n){switch(this._trackPointer(n),this._pointers.length){case 1:switch(this.touches.ONE){case Qi.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(n),this.state=at.TOUCH_ROTATE;break;case Qi.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(n),this.state=at.TOUCH_PAN;break;default:this.state=at.NONE}break;case 2:switch(this.touches.TWO){case Qi.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(n),this.state=at.TOUCH_DOLLY_PAN;break;case Qi.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(n),this.state=at.TOUCH_DOLLY_ROTATE;break;default:this.state=at.NONE}break;default:this.state=at.NONE}this.state!==at.NONE&&this.dispatchEvent(Jc)}function Nb(n){switch(this._trackPointer(n),this.state){case at.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(n),this.update();break;case at.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(n),this.update();break;case at.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(n),this.update();break;case at.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(n),this.update();break;default:this.state=at.NONE}}function Ub(n){this.enabled!==!1&&n.preventDefault()}function Ob(n){n.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function Fb(n){n.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}const kb=45,zb=600,oh=Math.PI/2+.02,Bb=Math.PI-.02;function Hb(n){return n<.5?4*n*n*n:1-(-2*n+2)**3/2}function ah(n){return{x:(n.x_min+n.x_max)/2,y:(n.y_min+n.y_max)/2}}function Gb(n){return Math.max(n.x_max-n.x_min,n.y_max-n.y_min,10)}class Vb{camera;controls;homeTarget=new R(150,0,-100);tween=null;constructor(e){this.camera=new sn(kb,e.clientWidth/Math.max(e.clientHeight,1),.5,8e3),this.controls=new Eb(this.camera,e),this.controls.enableDamping=!0,this.controls.dampingFactor=.12,this.controls.maxPolarAngle=oh,this.controls.minDistance=20,this.controls.maxDistance=4e3,this.camera.position.set(-160,320,420),this.controls.target.copy(this.homeTarget)}setHome(e){const t=ah(e),i=Pt(t,0);this.homeTarget.set(i.x,i.y,i.z),this.tween||this.controls.target.copy(this.homeTarget)}distanceToTarget(){return this.camera.position.distanceTo(this.controls.target)}preset(e,t){const i=Pt(ah(t),0),s=Gb(t)*1.55,r=new R(i.x,0,i.z);let o;switch(e){case"top":o=new R(i.x,s,i.z+s*.001);break;case"front":o=new R(i.x,s*.3,i.z+s);break;case"iso":o=new R(i.x-s*.65,s*.8,i.z+s*.9);break;default:o=this.camera.position.clone().sub(this.controls.target).normalize().multiplyScalar(s).add(r);break}this.animateTo(o,r)}animateTo(e,t){this.tween={fromPos:this.camera.position.clone(),toPos:e.clone(),fromTarget:this.controls.target.clone(),toTarget:t.clone(),start:performance.now()}}viewAxis(e){const t=new R(e.x,e.y,e.z);if(t.lengthSq()<1e-6)return;t.normalize(),this.controls.maxPolarAngle=t.y<-.999?Bb:oh;const i=this.controls.target.clone(),s=this.distanceToTarget(),r=i.clone().addScaledVector(t,s);Math.abs(t.y)>.999&&(r.z=i.z+s*.001),this.animateTo(r,i)}update(){if(this.tween){const e=Math.min(1,(performance.now()-this.tween.start)/zb),t=Hb(e);this.camera.position.lerpVectors(this.tween.fromPos,this.tween.toPos,t),this.controls.target.lerpVectors(this.tween.fromTarget,this.tween.toTarget,t),e>=1&&(this.tween=null)}this.controls.update()}resize(e,t){this.camera.aspect=e/Math.max(t,1),this.camera.updateProjectionMatrix()}dispose(){this.controls.dispose()}}function Kc(n){switch(n){case"X":return{x:1,y:0,z:0};case"-X":return{x:-1,y:0,z:0};case"Y":return{x:0,y:0,z:-1};case"-Y":return{x:0,y:0,z:1};case"Z":return{x:0,y:1,z:0};case"-Z":return{x:0,y:-1,z:0}}}const Rd=["X","-X","Y","-Y","Z","-Z"];function Wb(n){const{x:e,y:t,z:i,w:s}=n,r=c=>c===0?0:c,o={x:r(1-2*(t*t+i*i)),y:r(2*(e*t+i*s)),z:r(2*(e*i-t*s))},a={x:r(2*(e*t-i*s)),y:r(1-2*(e*e+i*i)),z:r(2*(t*i+e*s))},l={x:r(-2*(e*i+t*s)),y:r(-2*(t*i-e*s)),z:r(-(1-2*(e*e+t*t)))};return{right:o,up:a,forward:l}}function Xb(n,e){const t=(i,s)=>i.x*s.x+i.y*s.y+i.z*s.z;return Rd.map(i=>{const s=Kc(i);return{tip:i,x:e.centerX+e.radius*t(s,n.right),y:e.centerY-e.radius*t(s,n.up),depth:t(s,n.forward)}})}function $b(n,e,t,i){const s=t.map(o=>({t:o,dist:Math.hypot(n-o.x,e-o.y)})).filter(o=>o.dist<=i.tipRadius).sort((o,a)=>o.dist-a.dist);if(s.length>0){const o=s[0].dist;return s.filter(l=>l.dist<=o+1).sort((l,c)=>l.t.depth-c.t.depth)[0].t.tip}return Math.hypot(n-i.centerX,e-i.centerY)<=i.centerRadius?"center":null}const vi=64,Is=2.2,xi=Is*2/vi,Sa=46,Vr=13,Yb=14,qb=9,jb=3,ch=16,Zb=5,Jb=10,Kb=10,Qb=22,eM=46,tM={X:15026253,Y:5087231,Z:4630360};function nM(n){return new R(n.x,n.y,n.z)}function iM(){const e=document.createElement("canvas");e.width=256,e.height=256;const t=e.getContext("2d"),i=256/2;t.beginPath(),t.arc(i,i,i-8,0,Math.PI*2),t.fillStyle="rgba(30, 32, 38, 0.92)",t.fill();const s=new tr(e);return s.colorSpace=Ot,s}function sM(n,e,t){const s=document.createElement("canvas");s.width=64,s.height=64;const r=s.getContext("2d"),o=64/2;r.beginPath(),r.arc(o,o,27,0,Math.PI*2),r.fillStyle=t?"rgba(74, 78, 86, 0.9)":e,r.fill(),r.lineWidth=3,r.strokeStyle="rgba(0, 0, 0, 0.35)",r.stroke(),t||(r.fillStyle="#ffffff",r.font="bold 36px system-ui, sans-serif",r.textAlign="center",r.textBaseline="middle",r.fillText(n,o,o+2));const a=new tr(s);return a.colorSpace=Ot,a}function rM(){const e=document.createElement("canvas");e.width=32,e.height=32;const t=e.getContext("2d"),i=32/2;t.beginPath(),t.arc(i,i,9,0,Math.PI*2),t.fillStyle="rgba(154, 160, 170, 0.95)",t.fill();const s=new tr(e);return s.colorSpace=Ot,s}function oM(){const e=document.createElement("canvas");e.width=64,e.height=64;const t=e.getContext("2d"),i=64*.62,s=(64-i)/2;t.strokeStyle="#9aa0aa",t.fillStyle="#9aa0aa",t.lineWidth=4,t.lineJoin="round",t.beginPath(),t.moveTo(s,i*.5+s*.4),t.lineTo(64/2,s*.7),t.lineTo(64-s,i*.5+s*.4),t.stroke(),t.beginPath(),t.rect(64/2-i*.22,i*.52+s*.3,i*.44,i*.4),t.fill();const r=new tr(e);return r.colorSpace=Ot,r}function Wr(n,e){const t=new Vc(new Gc({map:n,transparent:!0,depthWrite:!1,depthTest:!1}));return t.scale.setScalar(e*xi),t}class aM{scene=new od;camera=new jc(-Is,Is,Is,-Is,.1,20);tipVisuals=[];house;offsetX=Qb;offsetY=eM;hover=null;constructor(){this.camera.position.set(0,0,5);const e=Wr(iM(),Sa+4);e.renderOrder=-2,this.scene.add(e);for(const i of Rd)this.scene.add(...this.buildTip(i));const t=Wr(rM(),jb);t.renderOrder=5,this.scene.add(t),this.house=Wr(oM(),ch),this.house.renderOrder=6,this.scene.add(this.house)}placeBadgeCorner(e){const t=Sa/2*xi,i=1*xi;e.position.set(-t+i,t-i,0)}buildTip(e){const t=e.replace("-",""),i=e.startsWith("-"),s=new Qe(tM[t]);i&&s.multiplyScalar(.55);const r=nM(Kc(e)),o=new qt({color:s,transparent:!0,opacity:i?.35:.9}),a=new xt(new So(.05,.05,Vr*xi,8),o);a.position.copy(r).multiplyScalar(Vr*xi/2),a.quaternion.setFromUnitVectors(new R(0,1,0),r.clone()),this.scene.add(a);const l=Wr(sM(t,`#${s.getHexString()}`,i),i?qb:Yb);return l.userData.baseScale=l.scale.x,l.position.copy(r).multiplyScalar(Vr*xi),l.renderOrder=i?2:4,this.scene.add(l),this.tipVisuals.push({tip:e,ball:l,baseColor:s}),[a,l]}setScreenOffset(e,t){this.offsetX=e,this.offsetY=t}setHover(e){if(e!==this.hover){this.hover=e;for(const t of this.tipVisuals){const i=t.tip===e;t.ball.scale.setScalar(t.ball.userData.baseScale*(i?1.25:1))}this.house.scale.setScalar(ch*xi*(e==="home"?1.3:1))}}geometry(e){const t=vi/2;return{centerX:e-this.offsetX-t,centerY:this.offsetY+t,radius:Vr}}hitTest(e,t,i,s){const r=e-s.left,o=t-s.top,a=this.geometry(s.width),l=Sa/2,c=a.centerX-l+5,u=a.centerY-l+5;if(Math.hypot(r-c,o-u)<=Kb)return"home";const h={...a,tipRadius:Jb,centerRadius:Zb};return $b(r,o,Xb(Wb(i.quaternion),a),h)}render(e,t){const i=e.getSize(new ue),s=vi/2,r=this.geometry(i.x);if(r.centerX-s<0||r.centerX+s>i.x||r.centerY-s<0||r.centerY+s>i.y)return;this.placeBadgeCorner(this.house),this.camera.quaternion.copy(t.quaternion),this.camera.position.set(0,0,5).applyQuaternion(this.camera.quaternion);const o=e.autoClear;e.autoClear=!1,e.setScissorTest(!0),e.setScissor(r.centerX-s,i.y-r.centerY-s,vi,vi),e.setViewport(r.centerX-s,i.y-r.centerY-s,vi,vi),e.clearDepth(),e.render(this.scene,this.camera),e.autoClear=o,e.setScissorTest(!1),e.setViewport(0,0,i.x,i.y)}dispose(){this.scene.traverse(e=>{if(e instanceof xt&&e.geometry.dispose(),e instanceof xt||e instanceof Vc){const t=e.material;t.map?.dispose(),t.dispose()}})}}const cM={r:.18,g:.55,b:1},lh={r:.72,g:.92,b:.2},lM={r:1,g:.15,b:.1};function uM(n){return Math.min(Math.max(n,0),1)}function uh(n,e,t){return t<=0?{...n}:t>=1?{...e}:{r:n.r+(e.r-n.r)*t,g:n.g+(e.g-n.g)*t,b:n.b+(e.b-n.b)*t}}function hM(n){const e=uM(n);return e<=.5?uh(cM,lh,e*2):uh(lh,lM,(e-.5)*2)}const Xr=.4,dM=10133674,fM=.32,pM=.95;class mM{group=new Ft;data=null;built=null;travelVisible=!0;getWorldToPanel;constructor(e){this.getWorldToPanel=e,this.group.visible=!1}setData(e){this.data&&e&&this.data.generationId===e.generationId||(this.data=e,this.rebuild())}viewportChanged(){this.data&&this.rebuild()}setVisible(e){this.group.visible=e&&this.data!==null}setTravelVisible(e){this.travelVisible=e,this.built&&(this.built.travelLines.visible=e)}rebuild(){this.disposeBuild(),this.group.visible=!1,!(!this.data||this.data.segments.length===0)&&(this.built=this.build(this.data.segments),this.built.travelLines.visible=this.travelVisible,this.group.add(this.built.cutLines,this.built.travelLines),this.group.visible=!0)}build(e){const t=this.getWorldToPanel(),i=[],s=[],r=[];for(const u of e){const h=Ta(t,u.points);if(u.cutting){const d=hM(u.power);for(let f=2;f+1<h.length;f+=2){const g=Pt({x:h[f-2],y:h[f-1]},Xr),_=Pt({x:h[f],y:h[f+1]},Xr);i.push(g.x,g.y,g.z,_.x,_.y,_.z),s.push(d.r,d.g,d.b,d.r,d.g,d.b)}}else for(let d=2;d+1<h.length;d+=2){const f=Pt({x:h[d-2],y:h[d-1]},Xr),g=Pt({x:h[d],y:h[d+1]},Xr);r.push(f.x,f.y,f.z,g.x,g.y,g.z)}}const o=new it;o.setAttribute("position",new gt(i,3)),o.setAttribute("color",new gt(s,3));const a=new ho(o,new Pn({vertexColors:!0,transparent:!0,opacity:pM,depthWrite:!1}));a.renderOrder=3;const l=new it;l.setAttribute("position",new gt(r,3));const c=new ho(l,new xd({color:dM,transparent:!0,opacity:fM,dashSize:1.6,gapSize:1.6,depthWrite:!1}));return c.computeLineDistances(),c.renderOrder=2,{cutLines:a,travelLines:c}}disposeBuild(){if(this.built){for(const e of[this.built.cutLines,this.built.travelLines])this.group.remove(e),e.geometry.dispose(),e.material.dispose();this.built=null}}dispose(){this.data=null,this.disposeBuild()}}const gM=.5,hh=24,_M=1e-4;class vM{group=new Ft;core;halo;haloMaterial;trail;trailPositions;trailGeometry;trailCount=0;last=null;constructor(){this.core=new xt(new $s(1.5,24),new qt({color:16777215,transparent:!0,opacity:.95,depthWrite:!1})),this.core.rotation.x=-Math.PI/2,this.core.renderOrder=7,this.group.add(this.core),this.haloMaterial=new qt({color:ks,transparent:!0,opacity:.4,depthWrite:!1}),this.halo=new xt(new $s(3.6,24),this.haloMaterial),this.halo.rotation.x=-Math.PI/2,this.halo.renderOrder=6,this.group.add(this.halo),this.trailPositions=new Float32Array(hh*3),this.trailGeometry=new it,this.trailGeometry.setAttribute("position",new hn(this.trailPositions,3)),this.trailGeometry.setDrawRange(0,0),this.trail=new Ln(this.trailGeometry,new Pn({color:ks,transparent:!0,opacity:.6,depthWrite:!1})),this.trail.renderOrder=5,this.group.add(this.trail),this.group.visible=!1}setPosition(e){if(!e){this.group.visible=!1,this.reset();return}const t=Pt(e,gM);this.group.visible=!0,this.core.position.set(t.x,t.y,t.z),this.halo.position.set(t.x,t.y,t.z),(!this.last||(this.last.x-t.x)**2+(this.last.y-t.y)**2>_M)&&(this.pushTrail(t),this.last={...t})}tick(e){this.group.visible&&(this.haloMaterial.opacity=.3+.18*Math.sin(e*8))}pushTrail(e){this.trailCount<hh?this.trailCount++:this.trailPositions.copyWithin(0,3);const t=(this.trailCount-1)*3;this.trailPositions[t]=e.x,this.trailPositions[t+1]=e.y,this.trailPositions[t+2]=e.z,this.trailGeometry.attributes.position.needsUpdate=!0,this.trailGeometry.setDrawRange(0,this.trailCount)}reset(){this.trailCount=0,this.trailGeometry.setDrawRange(0,0),this.last=null}dispose(){this.core.geometry.dispose(),this.core.material.dispose(),this.halo.geometry.dispose(),this.haloMaterial.dispose(),this.trailGeometry.dispose(),this.trail.material.dispose()}}const po=1,xM=15,Pd=[[0,0],[1,0],[1,1],[0,1]];function yM(n){return{pos:{x:n.pos_mm.x,y:n.pos_mm.y},size:{w:n.size_mm.w,h:n.size_mm.h},angleDeg:n.angle_deg}}function ba(n){return{pos_mm:{x:n.pos.x,y:n.pos.y},size_mm:{w:n.size.w,h:n.size.h},angle_deg:n.angleDeg}}function ci(n,e){const t=n*Math.PI/180,i=Math.cos(t),s=Math.sin(t);return{x:i*e.x-s*e.y,y:s*e.x+i*e.y}}function SM(n,e){return ci(-n,e)}function Qc(n,e){return{x:n.x+e.x,y:n.y+e.y}}function Zs(n,e){return{x:n.x-e.x,y:n.y-e.y}}function wc(n,e){const[t,i]=Pd[e];return Qc(n.pos,ci(n.angleDeg,{x:t*n.size.w,y:i*n.size.h}))}function Js(n){return Qc(n.pos,ci(n.angleDeg,{x:n.size.w/2,y:n.size.h/2}))}function bM(n){const e=Math.PI*2;let t=n%e;return t<=-Math.PI&&(t+=e),t>Math.PI&&(t-=e),t}function MM(n,e){return Math.round(n/e)*e||0}function wM(n,e){const t=e.corner??2,[i,s]=Pd[t],r=wc(e.transform,(t+2)%4),o=SM(e.transform.angleDeg,Zs(n.pointer,r)),a=Math.max(po,Math.abs(o.x)),l=Math.max(po,Math.abs(o.y));return{pos:Zs(r,ci(e.transform.angleDeg,{x:(1-i)*a,y:(1-s)*l})),size:{w:a,h:l},angleDeg:e.transform.angleDeg}}function EM(n,e){const t=Js(e.transform),i=Math.atan2(e.pointer.y-t.y,e.pointer.x-t.x),s=Math.atan2(n.pointer.y-t.y,n.pointer.x-t.x);let r=e.transform.angleDeg+bM(s-i)*180/Math.PI;n.snap&&(r=MM(r,xM));const o=ci(r,{x:e.transform.size.w/2,y:e.transform.size.h/2});return{pos:Zs(t,o),size:{...e.transform.size},angleDeg:r}}function TM(n,e){const t=Zs(n.pointer,e.pointer);return{pos:Qc(e.transform.pos,t),size:{...e.transform.size},angleDeg:e.transform.angleDeg}}class AM{startState=null;currentTransform=null;get active(){return this.startState!==null}get kind(){return this.startState?.kind??null}get start(){return this.startState}get current(){return this.currentTransform}begin(e){this.startState=e,this.currentTransform={...e.transform}}update(e){const t=this.startState;if(!t)return null;const i=t.kind,s=i==="move"?TM(e,t):i==="scale"?wM(e,t):EM(e,t);return this.currentTransform=s,s}cancel(){this.startState=null,this.currentTransform=null}}function Ma(n,e){return{a:n.a*e.a+n.b*e.d,b:n.a*e.b+n.b*e.e,d:n.d*e.a+n.e*e.d,e:n.d*e.b+n.e*e.e}}function CM(n,e){return{x:n.a*e.x+n.b*e.y,y:n.d*e.x+n.e*e.y}}function RM(n){const e=n.a*n.e-n.b*n.d;return e===0?{a:1,b:0,d:0,e:1}:{a:n.e/e,b:-n.b/e,d:-n.d/e,e:n.a/e}}function PM(n){return{a:n[0],b:n[1],d:n[3],e:n[4]}}function LM(n,e,t){const i=Math.abs(n.size.w)<1e-9?po:n.size.w,s=Math.abs(n.size.h)<1e-9?po:n.size.h,r=ci(e.angleDeg,{x:1,y:0}),o=ci(n.angleDeg,{x:1,y:0}),a={a:e.size.w/i*r.x,b:-(e.size.h/s)*r.y,d:e.size.w/i*r.y,e:e.size.h/s*r.x},l={a:o.x,b:o.y,d:-o.y,e:o.x},c=Ma(a,l),u=PM(t),h=RM(u),d=Ma(Ma(u,c),h),f=ii(t,Js(n)),g=ii(t,Js(e)),_=Zs(g,CM(d,f));return{linear:d,translation:_}}const Ld=.2,DM=.3,IM="#3a3e46",NM=2567738,UM=1915486,OM=1908774,FM=4870236,kM=.1,zM=.13;function BM(n){const e=[];for(const i of n)for(let s=0;s+1<i.length;s+=2)e.push(new R(i[s],Ld+.02,-i[s+1]));const t=new it().setFromPoints(e);return new Ln(t,new Pn({color:16777215}))}function HM(n){const e=[];for(const i of n)for(let s=0;s+1<i.length;s+=2)e.push(new R(i[s],zM,-i[s+1]));const t=new it().setFromPoints(e);return new Ln(t,new Pn({color:FM}))}function GM(n){if(n.length<6)return null;const e=new Mo;e.moveTo(n[0],n[1]);for(let i=2;i+1<n.length;i+=2)e.lineTo(n[i],n[i+1]);const t=new xt(new nr(e),new qt({color:OM,transparent:!0,opacity:.85,side:cn}));return t.rotation.x=-Math.PI/2,t.position.y=kM,t}function VM(n){if(n.length<6)return null;const e=new Mo;e.moveTo(n[0],n[1]);for(let i=2;i+1<n.length;i+=2)e.lineTo(n[i],n[i+1]);const t=new xt(new nr(e),new qt({transparent:!0,opacity:.9,side:cn}));return t.rotation.x=-Math.PI/2,t.position.y=Ld,t}function WM(n){let e=1/0,t=1/0,i=-1/0,s=-1/0;for(const r of n)for(let o=0;o+1<r.length;o+=2)e=Math.min(e,r[o]),i=Math.max(i,r[o]),t=Math.min(t,r[o+1]),s=Math.max(s,r[o+1]);return{xMin:e,yMin:t,xMax:i,yMax:s}}class XM{constructor(e,t){this.api=e,this.getWorldToPanel=t,this.group.add(this.stockGroup)}api;getWorldToPanel;group=new Ft;stockGroup=new Ft;cache=new Map;stockCache=new Map;selected=new Set;hovered={uid:null};pending=new Set;lastTree=null;bakedMatrix=null;imagesVisible=!0;stockVisible=!0;async sync(e){this.lastTree=e;const t=this.getWorldToPanel();this.bakedMatrix=t;const i=new Map;for(const r of e.layers){const o=new Qe(r.visible?r.color:IM);for(const a of r.workpieces)i.set(a.uid,{wp:a,color:o,visible:r.visible})}for(const r of[...this.cache.keys()])i.has(r)||this.remove(r);const s=[];for(const[r,{wp:o,color:a,visible:l}]of i){const c=this.cache.get(r);if(c&&!this.entryStale(c,o)){c.baseColor.copy(a),c.visible=l,this.paint(r);continue}this.pending.has(r)||(this.pending.add(r),s.push(this.loadWorkpiece(r,o,a,l,t).finally(()=>this.pending.delete(r))))}await Promise.all(s),this.syncStock(e.stock_items??[],t)}reproject(){const e=this.getWorldToPanel();if(cp(e,this.bakedMatrix))return Promise.resolve();for(const i of[...this.cache.keys()])this.remove(i);for(const i of[...this.stockCache.keys()])this.removeStock(i);this.bakedMatrix=e;const t=this.lastTree;return t?this.sync(t):Promise.resolve()}setSelection(e){this.selected.clear();for(const t of e)this.selected.add(t);for(const t of this.cache.keys())this.paint(t)}setHover(e){if(this.hovered.uid===e)return;const t=this.hovered.uid;this.hovered.uid=e,t&&this.paint(t),e&&this.paint(e)}setImageVisible(e){this.imagesVisible=e;for(const t of this.cache.values())t.textureMesh&&(t.textureMesh.visible=e)}get imageVisible(){return this.imagesVisible}setStockVisible(e){this.stockVisible=e,this.stockGroup.visible=e}get stockShown(){return this.stockVisible}pick(e){let t=null;for(const[i,s]of this.cache)s.visible&&s.panelPolylines.some(r=>up(e,r))&&(t=i);return t}setPreview(e,t,i,s){const r=this.cache.get(e);if(!r)return;const{linear:o,translation:a}=LM(t,i,s);r.holder.matrix.set(o.a,0,-o.b,a.x,0,1,0,0,-o.d,0,o.e,-a.y,0,0,0,1),r.holder.matrixWorldNeedsUpdate=!0}clearPreview(e){const t=this.cache.get(e);t&&(t.holder.matrix.identity(),t.holder.matrixWorldNeedsUpdate=!0)}clearAllPreviews(){for(const e of this.cache.keys())this.clearPreview(e)}entryStale(e,t){return bf(e.geometryRevision,t)||e.transformRevision!==t.transform_revision}async loadWorkpiece(e,t,i,s,r){try{const o=await this.api.getItemGeometry(e);this.remove(e);const a=o.polylines.map(h=>Ta(r,h)),l=new Ft;l.matrixAutoUpdate=!1;const c={geometryRevision:t.geometry_revision,transformRevision:t.transform_revision,visible:s,baseColor:i.clone(),panelPolylines:a,holder:l,fillMesh:VM(a[0]??[]),outline:BM(a),textureMesh:null};this.cache.set(e,c),this.paint(e),this.group.add(l),c.fillMesh&&l.add(c.fillMesh),l.add(c.outline);const u=await this.api.getItemTexture(e,t.geometry_revision);u&&this.cache.get(e)===c&&this.addTexture(c,u)}catch{return}}addTexture(e,t){new dv().load(t,i=>{i.colorSpace=Ot;const s=WM(e.panelPolylines),r=new xt(new ms(s.xMax-s.xMin,s.yMax-s.yMin),new qt({map:i,transparent:!0}));r.rotation.x=-Math.PI/2,r.position.set((s.xMin+s.xMax)/2,DM,-(s.yMin+s.yMax)/2),r.renderOrder=1,r.visible=this.imagesVisible,e.textureMesh=r,e.holder.add(r)},void 0,()=>{})}paint(e){const t=this.cache.get(e);if(!t)return;const i=this.selected.has(e),s=this.hovered.uid===e&&!i,r=i?new Qe(pb):s?t.baseColor.clone().lerp(new Qe(16777215),mb):t.baseColor;t.outline.material.color.copy(r),t.fillMesh&&t.fillMesh.material.color.setHex(i?UM:NM)}remove(e){const t=this.cache.get(e);if(t){for(const i of[t.fillMesh,t.outline,t.textureMesh]){if(!i)continue;t.holder.remove(i);const s=i;s.geometry?.dispose();const r=s.material;Array.isArray(r)?r.forEach(o=>o.dispose()):r?.dispose()}this.group.remove(t.holder),this.cache.delete(e)}}syncStock(e,t){const i=new Map(e.map(s=>[s.uid,s]));for(const s of[...this.stockCache.keys()])i.has(s)||this.removeStock(s);for(const[s,r]of i){const o=this.stockCache.get(s);o&&o.geometryRevision===r.geometry_revision&&o.transformRevision===r.transform_revision||this.loadStock(s,r,t)}}async loadStock(e,t,i){try{const s=await this.api.getItemGeometry(e);this.removeStock(e);const r=s.polylines.map(c=>Ta(i,c)),o=new Ft;o.matrixAutoUpdate=!1;const a=GM(r[0]??[]);a&&o.add(a),o.add(HM(r));const l={geometryRevision:t.geometry_revision,transformRevision:t.transform_revision,panelPolylines:r,holder:o};this.stockCache.set(e,l),this.stockGroup.add(o)}catch{return}}removeStock(e){const t=this.stockCache.get(e);if(t){for(const i of[...t.holder.children]){t.holder.remove(i);const s=i;s.geometry?.dispose(),s.material?.dispose()}this.stockGroup.remove(t.holder),this.stockCache.delete(e)}}}const eo=.6,dh=8,$M=14,YM=9,fh=72,ph=5087231,qM=4054148,jM=5087231;class ZM{group=new Ft;cornerHandles=[];rotateHandle;ring;outline;visible=!1;constructor(){const e=new fo(1,12,8);for(let t=0;t<4;t++){const i=new xt(e,new qt({color:ph}));i.renderOrder=10,this.cornerHandles.push(i),this.group.add(i)}this.rotateHandle=new xt(new fo(1,12,8),new qt({color:qM})),this.rotateHandle.renderOrder=10,this.group.add(this.rotateHandle),this.ring=new Ln(new it,new Pn({color:jM,transparent:!0,opacity:.45})),this.ring.renderOrder=9,this.group.add(this.ring),this.outline=new Ln(new it,new Pn({color:ph,transparent:!0,opacity:.8})),this.outline.renderOrder=9,this.group.add(this.outline),this.group.visible=!1}show(e,t){const i=[0,1,2,3].map(u=>this.scenePoint(wc(e,u),t));this.cornerHandles.forEach((u,h)=>{const d=i[h];u.position.set(d.x,eo,d.z)});const s=Js(e),r=this.scenePoint(s,t),o=this.ringRadius(e,t);this.ring.geometry.dispose(),this.ring.geometry=JM(r,o);const a=ci(e.angleDeg,{x:e.size.w/2,y:e.size.h/2+dh}),l=this.scenePoint({x:s.x+a.x,y:s.y+a.y},t);this.rotateHandle.position.set(l.x,eo,l.z);const c=[];for(let u=0;u<=4;u++){const h=i[u%4];c.push(new R(h.x,eo,h.z))}this.outline.geometry.dispose(),this.outline.geometry=new it().setFromPoints(c),this.group.visible=!0,this.visible=!0}hide(){this.group.visible=!1,this.visible=!1}get shown(){return this.visible}pick(e,t,i,s){if(!this.visible)return null;const r=this.cornerHandles.map((c,u)=>({object:c,hit:{kind:"scale",corner:u}}));r.push({object:this.rotateHandle,hit:{kind:"rotate"}});const o=new R;let a=1/0,l=null;for(const c of r){o.copy(c.object.position).project(i);const u=s.left+(o.x+1)/2*s.width,h=s.top+(1-o.y)/2*s.height,d=Math.hypot(u-e,h-t);d<=$M&&d<a&&(a=d,l=c.hit)}return l}updatePixelScale(e,t){if(!this.visible)return;const i=e.position.distanceTo(this.group.position),s=2*Math.max(i,1)*Math.tan(e.fov/2*Math.PI/180)/Math.max(t,1),r=Math.max(s*YM,.5);for(const o of[...this.cornerHandles,this.rotateHandle])o.scale.setScalar(r)}ringRadius(e,t){const i=ii(t,Js(e)),s=ii(t,wc(e,2));return Math.hypot(s.x-i.x,s.y-i.y)+dh}scenePoint(e,t){const i=Pt(ii(t,e),0);return{x:i.x,z:i.z}}}function JM(n,e){const t=[];for(let i=0;i<=fh;i++){const s=i/fh*Math.PI*2;t.push(new R(n.x+e*Math.cos(s),eo,n.z+e*Math.sin(s)))}return new it().setFromPoints(t)}const KM=5,QM=3;class ew{deps;mode={tag:"idle"};hoverUid=null;constructor(e){this.deps=e,e.dom.addEventListener("pointerdown",this.onPointerDown),window.addEventListener("pointermove",this.onPointerMove),window.addEventListener("pointerup",this.onPointerUp),window.addEventListener("keydown",this.onKeyDown)}setSelection(e){if(this.mode.tag!=="drag"){if(e.length===1){const t=this.deps.transformOf(e[0]);if(t){this.deps.gizmo.show(t,this.deps.worldToPanel());return}}this.deps.gizmo.hide()}}get dragging(){return this.mode.tag==="drag"}frame(){this.deps.gizmo.shown&&this.deps.gizmo.updatePixelScale(this.deps.camera(),this.deps.canvasRect().height)}cancelDrag(){if(this.mode.tag!=="drag")return;const e=this.mode.controller;this.mode={tag:"idle"},this.deps.workpieces.clearAllPreviews();const t=e.start;t&&this.deps.gizmo.show(t.transform,this.deps.worldToPanel()),e.cancel(),this.deps.controls.enabled=!0,this.deps.callbacks.onPreview(null,null)}dispose(){this.deps.dom.removeEventListener("pointerdown",this.onPointerDown),window.removeEventListener("pointermove",this.onPointerMove),window.removeEventListener("pointerup",this.onPointerUp),window.removeEventListener("keydown",this.onKeyDown)}beginDrag(e,t,i,s,r){const o=this.deps.transformOf(e);if(!o)return!1;const a=this.deps.panelAt(i.clientX,i.clientY);if(!a)return!1;const l=xl(this.deps.worldToPanel(),a),c=this.deps.selection(),u=t==="move"?c.filter(f=>f!==e):[],h=new AM;h.begin({kind:t,transform:o,pointer:l,corner:r});const d=new Map([[e,o]]);for(const f of u){const g=this.deps.transformOf(f);g&&d.set(f,g)}return this.mode={tag:"drag",uid:e,others:u,controller:h,startTransforms:d,downAt:{x:i.clientX,y:i.clientY},moved:!1,hitUid:s},this.deps.controls.enabled=!1,this.setHover(null),!0}onPointerDown=e=>{if(e.button!==0||this.mode.tag==="drag")return;const t=this.deps.gizmo.pick(e.clientX,e.clientY,this.deps.camera(),this.deps.canvasRect());if(t){const r=this.deps.selection()[0];if(r&&(t.kind==="rotate"?this.beginDrag(r,"rotate",e,null):this.beginDrag(r,"scale",e,null,t.corner)))return}const i=this.deps.panelAt(e.clientX,e.clientY),s=i?this.deps.workpieces.pick(i):null;s&&!e.shiftKey&&(this.deps.selection().includes(s)||this.deps.callbacks.onDragSelect(s),this.beginDrag(s,"move",e,s))||(this.mode={tag:"press",downAt:{x:e.clientX,y:e.clientY},hitUid:s})};onPointerMove=e=>{if(this.mode.tag==="drag"){Math.hypot(e.clientX-this.mode.downAt.x,e.clientY-this.mode.downAt.y)>QM&&(this.mode.moved=!0);const{uid:s,others:r,controller:o,startTransforms:a}=this.mode,l=this.deps.panelAt(e.clientX,e.clientY);if(!l)return;const c=this.deps.worldToPanel(),u=xl(c,l),h=o.update({pointer:u,snap:e.shiftKey}),d=o.start;if(!h||!d)return;if(this.deps.workpieces.setPreview(s,d.transform,h,c),o.kind==="move"){const f=h.pos.x-d.transform.pos.x,g=h.pos.y-d.transform.pos.y;for(const _ of r){const m=a.get(_);m&&this.deps.workpieces.setPreview(_,m,{pos:{x:m.pos.x+f,y:m.pos.y+g},size:{...m.size},angleDeg:m.angleDeg},c)}}this.deps.gizmo.show(h,c),this.deps.callbacks.onPreview(s,ba(h));return}if(e.buttons!==0){this.setHover(null);return}const t=this.deps.panelAt(e.clientX,e.clientY),i=t?this.deps.workpieces.pick(t):null;this.setHover(i)};onPointerUp=e=>{if(e.button===0){if(this.mode.tag==="drag"){const{uid:t,others:i,controller:s,startTransforms:r,moved:o,hitUid:a}=this.mode,l=s.current,c=s.start?.transform??null;if(this.mode={tag:"idle"},this.deps.controls.enabled=!0,this.deps.callbacks.onPreview(null,null),!(o&&l!==null&&c!==null&&this.transformChanged(c,l))){this.deps.workpieces.clearAllPreviews(),c&&this.deps.gizmo.show(c,this.deps.worldToPanel()),this.deps.callbacks.onClick(a,e);return}const h=[{uid:t,transform:ba(l)}];if(s.kind==="move"&&c){const d=l.pos.x-c.pos.x,f=l.pos.y-c.pos.y;for(const g of i){const _=r.get(g);_&&h.push({uid:g,transform:ba({pos:{x:_.pos.x+d,y:_.pos.y+f},size:{..._.size},angleDeg:_.angleDeg})})}}this.deps.callbacks.onCommit(h);return}if(this.mode.tag==="press"){const{downAt:t,hitUid:i}=this.mode;this.mode={tag:"idle"},Math.hypot(e.clientX-t.x,e.clientY-t.y)<=KM&&this.deps.callbacks.onClick(i,e)}}};onKeyDown=e=>{e.key==="Escape"&&this.mode.tag==="drag"&&(e.preventDefault(),this.cancelDrag())};transformChanged(e,t){return Math.abs(e.pos.x-t.pos.x)>1e-6||Math.abs(e.pos.y-t.pos.y)>1e-6||Math.abs(e.size.w-t.size.w)>1e-6||Math.abs(e.size.h-t.size.h)>1e-6||Math.abs(e.angleDeg-t.angleDeg)>1e-6}setHover(e){e!==this.hoverUid&&(this.hoverUid=e,this.deps.workpieces.setHover(e))}}const tw=new Gn(new R(0,1,0),0),nw=700,mh=5,iw=46;class sw{constructor(e,t,i={}){this.callbacks=i,this.container=e,this.renderer=new sb({antialias:!0}),this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,2)),this.renderer.setClearColor(th),this.renderer.autoClear=!0,e.appendChild(this.renderer.domElement),this.scene.background=new Qe(th),this.scene.add(new _v(16777215,.9));const s=new gv(16777215,.7);s.position.set(-300,500,400),this.scene.add(s),this.workpieces=new XM(t,()=>this.viewport?this.viewport.world_to_panel:[1,0,0,0,1,0]),this.opsPreview=new mM(()=>this.viewport?this.viewport.world_to_panel:[1,0,0,0,1,0]),this.scene.add(this.bed.group),this.scene.add(this.workpieces.group),this.scene.add(this.opsPreview.group),this.scene.add(this.beam.group),this.scene.add(this.playhead.group),this.scene.add(this.selectionGizmo.group),this.renderer.domElement.addEventListener("pointerdown",this.onGizmoPointerDown),window.addEventListener("pointermove",this.onGizmoPointerMove),window.addEventListener("pointerup",this.onGizmoPointerUp),this.rig=new Vb(this.renderer.domElement),this.gizmo.setScreenOffset(this.gizmoOffsetX(),iw),this.interaction=new ew({dom:this.renderer.domElement,camera:()=>this.rig.camera,canvasRect:()=>this.renderer.domElement.getBoundingClientRect(),worldToPanel:()=>this.viewport?this.viewport.world_to_panel:[1,0,0,0,1,0],panelAt:(r,o)=>this.panelPointAt(r,o),workpieces:this.workpieces,gizmo:this.selectionGizmo,controls:this.rig.controls,selection:()=>this.selection,transformOf:r=>this.callbacks.getTransform?.(r)??null,callbacks:{onClick:(r,o)=>this.callbacks.onClick?.(r,o),onDragSelect:r=>this.callbacks.onDragSelect?.(r),onPreview:(r,o)=>this.callbacks.onPreview?.(r,o),onCommit:r=>this.callbacks.onCommit?.(r)}}),this.resizeObserver=new ResizeObserver(()=>this.resize()),this.resizeObserver.observe(e),this.resize(),this.renderer.setAnimationLoop(()=>this.tick())}callbacks;renderer;scene=new od;rig;bed=new bb;workpieces;beam=new Mb;opsPreview;playhead=new vM;gizmo=new aM;raycaster=new yv;resizeObserver;container;selectionGizmo=new ZM;interaction=null;selection=[];viewport=null;clock=new xv;frameCount=0;gizmoPress=null;gizmoCursor=!1;setViewport(e){if(this.viewport===e||!Number.isFinite(e.extents.width_mm)||!Number.isFinite(e.extents.height_mm)||e.extents.width_mm<=0||e.extents.height_mm<=0)return;const t=this.viewport===null;this.viewport=e,this.bed.build(e),this.workpieces.reproject(),this.opsPreview.viewportChanged(),this.rig.setHome(e.workarea),t&&this.rig.preset("iso",e.workarea)}async syncTree(e){await this.workpieces.sync(e)}setSelection(e){this.selection=[...e],this.workpieces.setSelection(e),this.interaction?.setSelection(e)}discardPreviews(){this.workpieces.clearAllPreviews()}get dragging(){return this.interaction!==null&&this.interaction.dragging}setBeam(e){this.beam.setPosition(e)}setOpsPreview(e){this.opsPreview.setData(e)}setOpsVisible(e){this.opsPreview.setVisible(e)}setTravelVisible(e){this.opsPreview.setTravelVisible(e)}setImageVisible(e){this.workpieces.setImageVisible(e)}setStockVisible(e){this.workpieces.setStockVisible(e)}setNogoZonesVisible(e){this.bed.setNogoZonesVisible(e)}setPlayhead(e){this.playhead.setPosition(e)}viewPreset(e){this.viewport&&this.rig.preset(e,this.viewport.workarea)}setNavOffset(e,t){this.gizmo.setScreenOffset(e,t)}gizmoOffsetX(){return 22}gizmoHitAt(e){return e.target!==this.renderer.domElement?null:this.gizmo.hitTest(e.clientX,e.clientY,this.rig.camera,this.renderer.domElement.getBoundingClientRect())}onGizmoPointerDown=e=>{if(e.button!==0)return;const t=this.gizmoHitAt(e);t&&(e.stopImmediatePropagation(),this.gizmoPress={x:e.clientX,y:e.clientY,hit:t},this.rig.controls.enabled=!1)};onGizmoPointerMove=e=>{if(this.gizmoPress){Math.hypot(e.clientX-this.gizmoPress.x,e.clientY-this.gizmoPress.y)>mh&&this.clearGizmoPress();return}const t=this.gizmoHitAt(e),i=t!==null&&t!=="center"?t:null;this.gizmo.setHover(i),this.setGizmoCursor(t!==null)};onGizmoPointerUp=e=>{const t=this.gizmoPress;t&&(this.clearGizmoPress(),!(Math.hypot(e.clientX-t.x,e.clientY-t.y)>mh)&&this.activateGizmoHit(t.hit))};clearGizmoPress(){this.gizmoPress=null,this.rig.controls.enabled=!0,this.setGizmoCursor(!1),this.gizmo.setHover(null)}setGizmoCursor(e){e!==this.gizmoCursor&&(this.gizmoCursor=e,this.renderer.domElement.style.cursor=e?"pointer":"")}activateGizmoHit(e){if(this.viewport){if(e==="center"||e==="home"){this.rig.preset("iso",this.viewport.workarea);return}this.rig.viewAxis(Kc(e))}}panelPointAt(e,t){const i=this.renderer.domElement.getBoundingClientRect(),s=new ue((e-i.left)/i.width*2-1,-((t-i.top)/i.height)*2+1);this.raycaster.setFromCamera(s,this.rig.camera);const r=new R;return this.raycaster.ray.intersectPlane(tw,r)?ap(r.x,r.y,r.z):null}resize(){const e=this.container.clientWidth,t=this.container.clientHeight;e===0||t===0||(this.renderer.setSize(e,t),this.rig.resize(e,t))}tick(){const e=this.clock.getElapsedTime();this.rig.update(),this.beam.tick(e),this.playhead.tick(e),this.interaction?.frame(),this.frameCount++,this.frameCount%15===0&&this.bed.setDensity(this.rig.distanceToTarget()<nw?"fine":"coarse"),this.renderer.render(this.scene,this.rig.camera),this.gizmo.render(this.renderer,this.rig.camera)}dispose(){this.renderer.setAnimationLoop(null),this.resizeObserver.disconnect(),this.renderer.domElement.removeEventListener("pointerdown",this.onGizmoPointerDown),window.removeEventListener("pointermove",this.onGizmoPointerMove),window.removeEventListener("pointerup",this.onGizmoPointerUp),this.gizmo.dispose(),this.interaction?.dispose(),this.opsPreview.dispose(),this.playhead.dispose(),this.rig.dispose(),this.renderer.dispose(),this.renderer.domElement.remove()}}const rw=300;function ow(){return{newDoc:()=>{},openProject:()=>{},saveProject:()=>{},exportGcode:()=>{},undo:()=>{},redo:()=>{},deleteSelected:()=>{},duplicateSelected:()=>{},flip:()=>{},align:()=>{},autoLayout:()=>{},addStep:()=>{},machineSettings:()=>{},openSettings:()=>{},importFile:()=>{},setTool:()=>{},viewPreset:()=>{},frameSelection:()=>{},zoom:()=>{},toggleToolpaths:()=>{},startJob:()=>{},homeAll:()=>{},setOriginHere:()=>{},goToOrigin:()=>{},refreshDoc:async()=>{},about:()=>{}}}function aw(n){const e=document.querySelector(".viewport-host canvas");e&&e.dispatchEvent(new WheelEvent("wheel",{deltaY:n,bubbles:!0,cancelable:!0}))}function cw(n,e=ow()){n.classList.add("app");const t=Tm({newDoc:e.newDoc,openProject:e.openProject,saveProject:e.saveProject,exportGcode:e.exportGcode,undo:e.undo,redo:e.redo,importFile:e.importFile,duplicateSelected:e.duplicateSelected,deleteSelected:e.deleteSelected,viewPreset:e.viewPreset,frameSelection:e.frameSelection,toggleToolpaths:e.toggleToolpaths,startJob:e.startJob,openSettings:e.openSettings,about:e.about}),i=yp({setTool:e.setTool,newDoc:e.newDoc,openProject:e.openProject,saveProject:e.saveProject,importFile:e.importFile,exportGcode:e.exportGcode,duplicateSelected:e.duplicateSelected,deleteSelected:e.deleteSelected,flip:e.flip,align:e.align,autoLayout:c=>e.autoLayout(c),addStep:()=>e.addStep(),machineSettings:()=>e.machineSettings(),frameSelection:e.frameSelection,viewPreset:e.viewPreset,zoom:e.zoom,startJob:e.startJob,homeAll:e.homeAll,setOriginHere:e.setOriginHere,goToOrigin:e.goToOrigin,about:e.about}),s=Bp({refreshDoc:e.refreshDoc}),r=_m({refreshDoc:e.refreshDoc,newDoc:e.newDoc,deleteSelected:e.deleteSelected,duplicateSelected:e.duplicateSelected,flip:e.flip,align:e.align}),o=ig(),a=E("div","viewport-host"),l=sg();return a.appendChild(l.root),a.appendChild(Gm()),a.appendChild(Bm()),n.append(a,t.root,i.root,s.root,s.stub,r.root,r.stub,o.root),n.appendChild(ag()),{root:n,viewportHost:a,topbar:t,toolbar:i,leftPanel:s,rightPanel:r,statusbar:o,transport:l}}function lw(n){let e=null,t=null;async function i(){try{_l(await le.getViewport())}catch(D){_e("error",`Viewport fetch failed: ${Ht(D)}`)}}async function s(){try{ml(await le.getDocTree()),await r()}catch(D){_e("error",`Document fetch failed: ${Ht(D)}`)}}async function r(){try{Af(await le.getDocHistory())}catch{return}}async function o(){try{Ea(await le.getMachineState())}catch{return}}async function a(){try{vl(await le.getJobStatus())}catch{return}}function l(){le.postNewDoc().then(()=>(Sl("Untitled"),s())).catch(D=>_e("error",`New document failed: ${Ht(D)}`))}function c(){const D=E("input");D.type="file",D.accept=".ryp",D.style.display="none",D.addEventListener("change",()=>{const W=D.files?.[0];D.remove(),W&&le.openProject(W).then(async Q=>{ml(Q),await r(),Sl(W.name),_e("success",`Opened ${W.name}`)}).catch(Q=>_e("error",`Opening project failed: ${Ht(Q)}`))}),document.body.appendChild(D),D.click()}function u(){le.saveProject().then(D=>{bl(D,"project.ryp"),_e("success","Project saved")}).catch(D=>_e("error",`Saving project failed: ${Ht(D)}`))}function h(){le.exportGcode().then(D=>{bl(new Blob([D],{type:"text/plain"}),"gcode.g"),_e("success","G-code exported")}).catch(D=>{D instanceof Ki&&D.status===404?_e("warning","No G-code generated yet"):_e("error",`G-code export failed: ${Ht(D)}`)})}function d(){T.undo()}function f(){T.redo()}let g=null;function _(){g&&clearTimeout(g),g=setTimeout(()=>{g=null,s()},rw)}function m(){g!==null&&(clearTimeout(g),g=null)}async function p(D){let W;try{W=await le.getLayouts()}catch(De){_e("error",`Loading layouts failed: ${Ht(De)}`);return}const Q=$m(W);if(Q.kind==="none"){_e("warning","No layout strategies available");return}const me=Q.kind==="run"?Q.layout:await qm(D,Q.layouts);if(me)try{await le.runLayout(me.id),await s(),_e("success",`${me.label} applied`)}catch(De){_e("error",`Auto layout failed: ${Ht(De)}`)}}function C(){const{layers:D}=We.getState();return D.find(W=>W.is_active)?.uid??D[0]?.uid??null}function M(){const D=C();if(!D){_e("warning","No layer to add a step to");return}const W=We.getState().layers.find(Q=>Q.uid===D);Ih({uid:D,name:W?.name??"Layer"},()=>s())}const x=[{label:"General",fields:[{key:"theme",label:"Theme",kind:"choice",options:[{value:"system",label:"System"},{value:"light",label:"Light"},{value:"dark",label:"Dark"}]},{key:"language",label:"Language",kind:"choice",options:[{value:"",label:"System default"},{value:"en",label:"English"},{value:"de",label:"Deutsch"},{value:"es",label:"Español"},{value:"fr",label:"Français"},{value:"hi",label:"हिन्दी"},{value:"pt",label:"Português"},{value:"uk",label:"Українська"},{value:"zh_CN",label:"中文"}]},{key:"check_for_app_updates",label:"Check for updates",kind:"switch"}],note:"Theme and language apply after reloading the page."},{label:"Units",fields:[{key:"unit_preferences.length",label:"Length",kind:"choice",options:[{value:"mm",label:"mm"},{value:"cm",label:"cm"},{value:"m",label:"m"},{value:"in",label:"in"},{value:"ft",label:"ft"}]},{key:"unit_preferences.speed",label:"Speed",kind:"choice",options:[{value:"mm/min",label:"mm/min"},{value:"mm/s",label:"mm/s"},{value:"in/min",label:"in/min"},{value:"in/s",label:"in/s"}]},{key:"unit_preferences.acceleration",label:"Acceleration",kind:"choice",options:[{value:"mm/s²",label:"mm/s²"},{value:"cm/s²",label:"cm/s²"},{value:"m/s²",label:"m/s²"},{value:"in/s²",label:"in/s²"},{value:"ft/s²",label:"ft/s²"}]}]},{label:"Import",fields:[{key:"import_dpi",label:"Import DPI",kind:"number"},{key:"watch_source_files",label:"Watch source files",kind:"switch"}]},{label:"Advanced",fields:[{key:"ops_color_mode",label:"Ops color source",kind:"choice",options:[{value:"laser",label:"Laser power"},{value:"layer",label:"Layer color"}]},{key:"cache_budget_bytes",label:"Cache budget (bytes)",kind:"number"}]}];function P(){jm({title:"App settings",sections:x,load:()=>le.getSettings(),save:D=>{const W={...D};return W.language===""&&(W.language=null),le.putSettings(W)},savedMessage:"Settings saved"})}function w(){const D=Et.getState().activeMachineId;if(!D){_e("warning","No active machine");return}eg({machineId:D,onSaved:()=>{i(),o()}})}const T=new Nf({api:le,refreshDoc:()=>s(),onError:D=>_e("error",D)}),L={async homeAll(){try{await le.homeMachine(),await o()}catch(D){_e("error",`Homing failed: ${Ht(D)}`)}},async setOriginHere(){try{await le.setWorkOriginHere(),await o(),await i()}catch(D){_e("error",`Set work origin failed: ${Ht(D)}`)}},async goToOrigin(){const D=Et.getState().viewport?.workarea_origin_panel;if(D)try{await le.moveTo(D.x,D.y),await o()}catch(W){_e("error",`Move failed: ${Ht(W)}`)}}},y=cw(n,{newDoc:l,openProject:c,saveProject:u,exportGcode:h,undo:d,redo:f,deleteSelected:()=>T.deleteSelected(),duplicateSelected:()=>T.duplicateSelected(),flip:D=>T.flipSelected(D),align:D=>T.alignSelected(D),autoLayout:D=>{p(D)},addStep:()=>M(),machineSettings:()=>{w()},importFile:()=>Nl({refreshDoc:()=>s()}),setTool:()=>{},viewPreset:D=>e?.viewPreset(D),frameSelection:()=>{le.frame().then(()=>_e("info","Framing the workpiece on the machine")).catch(D=>_e("error",`Frame failed: ${Ht(D)}`))},zoom:D=>aw(D),toggleToolpaths:()=>Ah(!je.getState().showToolpaths),startJob:()=>{le.startJob().catch(D=>_e("error",`Start failed: ${Ht(D)}`))},openSettings:()=>{P()},homeAll:()=>{L.homeAll()},setOriginHere:()=>{L.setOriginHere()},goToOrigin:()=>{L.goToOrigin()},refreshDoc:()=>s(),about:()=>{const D=je.getState().systemInfo;_e("info",D?`Rayforge ${D.version} · API ${D.api_version} · Python ${D.python_version}`:"Rayforge")}});try{e=new sw(y.viewportHost,le,{onClick:(D,W)=>{D?ei([D],W.shiftKey):Cf()},onDragSelect:D=>ei([D]),onPreview:(D,W)=>{if(!D||!W){Ff();return}Of({uid:D,pos:W.pos_mm,size:W.size_mm,angleDeg:W.angle_deg})},onCommit:D=>{N(D)},getTransform:D=>{const W=Tc(We.getState().layers,D);return W?yM(W.workpiece):null}}),y.viewportHost.classList.add("has-canvas")}catch(D){_e("error",`WebGL unavailable: ${Ht(D)}`)}y.viewportHost.appendChild(Xm({api:le,onRollStart:()=>{},onRollEnd:()=>{m(),s()},onError:D=>_e("error",D)}).root);function b(){const D=je.getState();e?.setTravelVisible(D.showTravelMoves),e?.setImageVisible(D.showWorkpieceImages),e?.setNogoZonesVisible(D.showNogoZones),e?.setStockVisible(D.showStock)}b(),je.subscribe(b);function N(D){Promise.all(D.map(W=>le.putItemTransform(W.uid,W.transform))).then(()=>s()).catch(W=>{e?.discardPreviews(),_e("error",`Transform failed: ${Ht(W)}`)})}let O=null;We.subscribe(D=>{e?.setSelection(D.selection),D.layers!==O&&(O=D.layers,e?.syncTree({rev:D.rev,layers:D.layers,assets_count:D.assetsCount,stock_items:D.stocks}))});function G(D,W){const Q=Et.getState().viewport?.world_to_panel??null,me=Vf(D,W);return!me||!Q?null:ii(Q,{x:me.x,y:me.y})}function V(){const D=lt.getState(),W=Yt.getState(),Q=Et.getState();Q.viewport&&e?.setViewport(Q.viewport),e?.setBeam(Q.machineState?.beam_position_panel??Q.viewport?.beam_position_panel??null),e?.setOpsPreview(D.available?{generationId:D.generationId??"",segments:D.segments}:null),e?.setOpsVisible(je.getState().showToolpaths&&D.available);const me=D.track;let De=null;me&&(W.active?De=G(me,W.progressFraction*me.totalDistance):(D.playing||D.scrubbing)&&(De=G(me,D.positionMm),e?.setBeam(null))),e?.setPlayhead(De)}Et.subscribe(V),Yt.subscribe(V),lt.subscribe(V),je.subscribe(D=>{e?.setOpsVisible(D.showToolpaths&&lt.getState().available)}),je.subscribe(D=>{const W=y.topbar.connectionPill,Q=y.topbar.connectionText;W.classList.toggle("connected",D.wsStatus==="open"),W.classList.toggle("disconnected",D.wsStatus!=="open");const me=D.mockMode?" (mock)":"";Q.textContent=D.wsStatus==="open"?`Live${me}`:`Offline${me}`}),y.topbar.connectionPill.addEventListener("click",()=>{_e("info",`Event stream: ${je.getState().wsStatus}`)});const X=D=>{if(T.handleKeydown(D),D.defaultPrevented||!(D.ctrlKey||D.metaKey))return;const W=D.key.toLowerCase();W==="n"?(D.preventDefault(),l()):W==="s"?(D.preventDefault(),u()):W==="o"?(D.preventDefault(),c()):W==="i"&&(D.preventDefault(),Nl({refreshDoc:()=>s()}))};window.addEventListener("keydown",X);function q(D){_p(D,{onDoc:W=>{const Q=Sf(W,We.getState().rev);We.getState().rolling&&Q.refetchTree?_():Q.refetchTree?s():r()},onViewport:W=>_l(Sh(W)),onMachine:W=>Ea(bh(W)),onJob:W=>{const Q=Mh(W),me=Yt.getState().active;vl(Q),Jf(me,Q.active)&&Ro()},onNotify:(W,Q)=>{_e(W==="error"||W==="warning"||W==="success"?W:"info",Q)}})}function J(){if(je.getState().mockMode){t=Qd(q),yl("open");return}t=gp(le.getWsUrl(),{onEvent:q,onStatus:yl},()=>{i(),s(),o(),a(),Ro()})}async function z(){try{dp(await le.getSystemInfo())}catch(D){_e("warning",`Backend not reachable: ${Ht(D)}`)}await i(),await s();try{kf(await le.getMachines())}catch{_e("warning","Machine list unavailable")}await o(),await a(),await Ro(),J()}const H=op();return z(),{destroy(){H(),m(),t?.close(),window.removeEventListener("keydown",X),e?.dispose(),n.classList.remove("app"),n.replaceChildren()}}}function Ht(n){return n instanceof Error?n.message:String(n)}const gh=document.getElementById("app");gh&&lw(gh);
