'use strict';

const $ = id => document.getElementById(id);
const h = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const valid = value => value !== null && value !== undefined && Number.isFinite(Number(value));
const n = (value, digits=0) => valid(value) ? Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits,minimumFractionDigits:digits}) : '—';
const pct = value => valid(value) ? n(Number(value)*100,1)+'%' : '—';
function duration(value) {
  if (!valid(value)) return '尚无记录';
  let seconds = Math.max(0, Math.round(value));
  const hours = Math.floor(seconds/3600); seconds %= 3600;
  const minutes = Math.floor(seconds/60); seconds %= 60;
  return (hours ? hours+'小时' : '') + (minutes ? minutes+'分' : '') + (seconds || (!hours && !minutes) ? seconds+'秒' : '');
}
function date(value, full=false) {
  if (!value || !Number.isFinite(new Date(value).getTime())) return '—';
  return new Date(value).toLocaleString('zh-CN', full ? {} : {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
}
const modeName = m => m==='privileged' ? '特权教师 · 无雷达' : '雷达策略';
const reasonNames = {collision:'碰撞',stuck:'卡住',looping:'局部绕圈',slow:'持续低速',out_of_bounds:'越界',time_limit:'正常达到时限'};
const statusNames = {training:'训练中',starting:'启动中',stopped:'已主动停止',complete:'已完成',time_budget_reached:'时间预算已用完',interrupted:'已中断',failed:'异常退出',stale:'数据中断，状态待确认'};
const state = {data:null,config:null,runId:'latest',tab:'overview',chart:'duration',episodeSource:'training',reason:'all',limit:20,busy:false,error:null,fallback:false,chartPoints:[]};
const chartSpecs = {
  duration:['巡航时长','回合平均持续时间','秒',1], reward:['回合奖励','原始回合奖励','',1],
  completion:['完成率','正常达到时限的比例','%',100],collision:['碰撞率','训练碰撞率','%',100],
  stuck:['卡住率','训练卡住率','%',100],speed:['实际速度','回合平均前进速度','m/s',1],
};
const run = () => (state.data?.runs || []).find(r=>r.id === (state.runId==='latest' ? state.data.default_run_id : state.runId));
const fresh = stamp => stamp && Date.now()-new Date(stamp).getTime() <= (state.data?.stale_after_sec || 180)*1000;
const workerName = name => /^worker_\d+$/.test(name || '') ? '训练车 '+(Number(name.split('_')[1])+1) : (name || '独立验收');
const badge = (text, style='') => `<span class="badge ${style}">${h(text)}</span>`;
const metric = (label,value,caption='') => `<div class="metric"><div class="label">${h(label)}</div><strong>${h(value)}</strong><div class="caption">${h(caption)}</div></div>`;
const detailList = rows => `<dl class="detail-list">${rows.map(([k,v])=>`<div><dt>${h(k)}</dt><dd>${h(v ?? '—')}</dd></div>`).join('')}</dl>`;

function renderFreshness() {
  if (!state.data) return;
  const r=run(), evaluations=state.data.evaluations || [];
  const active = [r?.is_running ? r : null,...evaluations.filter(e=>e.is_running)].filter(Boolean);
  const stale = active.some(item=>!fresh(item.updated_at)) || r?.status==='stale' || evaluations.some(e=>e.status==='stale');
  const b=$('connection-badge');
  b.className='badge'+(state.error || state.fallback || stale ? ' warn' : '');
  b.textContent=state.error ? '连接中断' : state.fallback ? '备用快照' : stale ? '数据已过期' : active.length ? '数据同步中' : '历史结果';
  const source=active.length ? active.map(x=>x.updated_at).sort((a,b)=>new Date(b)-new Date(a))[0] : state.data.generated_at;
  const age=Math.max(0,(Date.now()-new Date(source).getTime())/1000);
  $('data-age').textContent=active.length ? `最近数据 · ${duration(age)}前` : stale ? '运行状态待确认，当前显示最后记录' : '当前没有正在运行的训练或评估';
  let message=state.error ? '暂时无法取得新数据，已保留上一次快照。请检查电脑端进度服务和网络。' : state.fallback ? '实时数据源暂时不可达，当前显示网页部署时的备用快照，请留意下方快照时间。' : stale ? '训练数据长时间没有更新，当前状态待确认；显示的是最后记录，请检查电脑端训练和上传进程。' : '';
  $('notice').hidden=!message; $('notice').textContent=message;
  $('snapshot-time').textContent='快照生成：'+date(state.data.generated_at,true)+'（不是浏览器刷新时间）';
  $('update-policy').textContent=state.config.mode==='github' ? `电脑端约 ${state.config.publish_seconds} 秒同步 · 手机每 ${state.config.poll_seconds} 秒检查 · GitHub 缓存或网络可能增加延迟` : `本地每 ${state.config.poll_seconds} 秒更新 · 时间与距离以仿真实测为准`;
}

function renderOverview(r) {
  const model=state.data.validated_model, s=r?.recent;
  const matched=model?.verified && model.training_run===r?.id;
  const evaluating=(state.data.evaluations || []).find(e=>e.is_running);
  let title, value, subtitle, status;
  if(r?.is_running || r?.status==='stale') {
    const workers=(r.workers || []).filter(w=>w.phase==='driving'||w.phase==='ready');
    const elapsed=workers.length ? Math.max(...workers.map(w=>w.sim_duration_sec || 0)) : null;
    title=r.phase==='periodic_evaluation' ? '训练暂停采样 · 正在定期评估' : '当前训练回合 · 最长已行驶';
    value=duration(elapsed); subtitle='各车独立回合计时；复位后重新开始，不跨回合累加。';
    status=fresh(r.updated_at) ? (statusNames[r.status] || '训练中') : '最后记录 · 状态待确认';
  } else if(evaluating) {
    title='独立评估 · 当前这一段';value=duration(evaluating.live?.sim_duration_sec);
    subtitle=`已结束 ${evaluating.completed_episodes} / ${evaluating.total_episodes} 段 · ${evaluating.stats.failure_count} 次失败`;
    status=fresh(evaluating.updated_at) ? '正在评估固定权重' : '最后记录 · 状态待确认';
  } else if(matched) {
    title='已验证连续巡航 · 单段至少'; value=duration(model.min_completed_sec);
    subtitle=`${model.stats.count} 段独立验收 · 每段达到时限后复位 · ${model.stats.failure_count} 次失败`;
    status='模型已验证';
  } else {
    title='选中批次 · 最长完整巡航';value=duration(s?.longest_completed_sec);
    subtitle='训练探索回合的记录；更多驾驶结论请查看独立验收。';status=statusNames[r?.status] || '尚无训练记录';
  }
  const start=r?.start_steps || 0, goal=r?.target_steps;
  const stepProgress=valid(goal)&&goal>start ? 100*((r?.steps||0)-start)/(goal-start) : 0;
  const budget=r?.time_budget_sec, timed=valid(budget)&&budget>0;
  const progress=Math.min(100,Math.max(0,stepProgress,timed ? 100*(r?.elapsed_wall_sec||0)/budget : 0));
  const progressLabel=timed ? `时间预算 ${duration(budget)} · 已用 ${duration(r?.elapsed_wall_sec)}` : `${n(r?.steps)} / ${n(goal)} 步`;
  const budgetNote=r?.is_running && valid(r?.remaining_wall_sec) ? `剩余约 ${duration(r.remaining_wall_sec)} · 含定期评估` : '电脑时间，含定期评估';
  $('hero').className='hero';
  $('hero').innerHTML=`<p class="eyebrow">${h(title)}</p><div class="hero-value">${h(value)}</div><p>${h(subtitle)}</p><div class="hero-status">${badge(status)}<span class="caption" style="color:#c1d2d9">${h(progressLabel)}</span></div><div class="progress-track" aria-label="本轮训练时间或采样预算进度"><i style="width:${progress}%"></i></div>`;
  $('headline-metrics').innerHTML=metric('累计训练采样',n(r?.steps),'包含续训前的步数')+metric('本轮实际耗时',duration(r?.elapsed_wall_sec),budgetNote)+metric('近期完成率',pct(s?.completion_rate),`${s?.count || 0} 个已结束训练回合`)+metric('平均前进速度',n(s?.mean_speed_mps,4),'m/s · 目标 '+n(r?.vehicle?.forward_speed_mps,2));
  $('window-label').textContent=`最近 ${s?.count || 0} 个训练回合`;
  $('failure-cards').innerHTML=['collision','stuck'].map(key=>{
    const f=s?.failures?.[key], count=f?.count || 0;
    return `<article class="card failure-card"><h3>${h(reasonNames[key])}<span class="count">${count} 次 / ${s?.count || 0} 回合</span></h3><strong>${count ? h(duration(f.mean_sec)) : '尚未观察到'}</strong><p>${count ? '仅发生该失败的回合：平均发生时间' : '此窗口内未发生，不能推算平均时间'}</p>${count ? `<p>最近一次：${h(duration(f.last.sim_duration_sec))} · ${n(f.last.distance_m,2)} m</p><p>最早发生：${h(duration(f.min_sec))}</p>` : ''}</article>`;
  }).join('')+`<div class="minor-failures"><span>局部绕圈 ${s?.counts?.looping||0}</span><span>低速 ${s?.counts?.slow||0}</span><span>越界 ${s?.counts?.out_of_bounds||0}</span><span>正常达到时限 ${s?.counts?.time_limit||0}</span></div>`;
  const workers=r?.workers || [];
  const liveWorkers=r?.is_running || r?.status==='stale' ? [...workers] : [];
  if(r?.phase==='periodic_evaluation' && r.periodic_evaluation) liveWorkers.push(r.periodic_evaluation);
  $('live-section').hidden=!liveWorkers.length;
  $('live-cards').innerHTML=liveWorkers.map(liveCard).join('');
  const evaluations=[...(state.data.evaluations || [])].sort((a,b)=>Number(b.is_running)-Number(a.is_running)).slice(0,2);
  $('evaluation-section').hidden=!evaluations.length;
  $('evaluation-cards').innerHTML=evaluations.map(e=>`<div class="card"><div class="live-head"><h3>${e.is_running ? '固定权重独立评估中' : '评估：'+h(statusNames[e.status] || e.status)}</h3>${badge(`${e.completed_episodes}/${e.total_episodes} 段`,e.is_running || e.passed?'':'warn')}</div><p class="caption">模型 ${h((e.model_sha256 || '').slice(0,12))} · ${e.stats.failure_count} 次失败 · 正常完成 ${e.stats.counts.time_limit} 段</p>${e.live ? `<div class="live-duration">${h(duration(e.live.sim_duration_sec))}</div><p class="caption">${e.is_running?'当前这一段':'最后一段记录'} · ${n(e.live.distance_m,2)} m · ${n(e.live.speed_mps,4)} m/s</p>` : '<p class="caption">尚无逐步记录</p>'}<p class="caption">${['collision','stuck'].map(k=>h(reasonNames[k])+': '+(e.stats.failures[k].count ? n(e.stats.failures[k].count)+' 次，平均 '+h(duration(e.stats.failures[k].mean_sec))+' 时发生' : '尚未观察到')).join('<br>')}</p></div>`).join('');
  $('validation-card').innerHTML=model ? `<div class="validation-head"><h3>${h(modeName(model.observation_mode))} · ${n(model.steps)} 步</h3>${badge(model.verified?'散列与报告匹配':'权重或报告不匹配',model.verified?'':'danger')}</div><div class="validation-value">${n(model.stats.failure_count)} 次失败<span>/ ${model.stats.count} 段独立验收</span></div><p class="caption">最短单段 ${h(duration(model.min_completed_sec))} · 平均速度 ${n(model.min_mean_speed_mps,5)}～${n(model.max_mean_speed_mps,5)} m/s</p><div class="validation-foot"><span>累计 ${n(model.stats.total_distance_m,1)} m</span><span>累计仿真 ${h(duration(model.stats.total_sim_sec))}</span><span>最低速度合规率 ${pct(model.min_speed_compliance)}</span></div><p class="caption">独立验收与上面的随机探索训练分开统计；累计时间是多段之和。</p>` : '<div class="empty">还没有已冻结的验证模型。训练检查点不会自动当作通过验收。</div>';
}

function liveCard(w) {
  const phases={starting:'启动中',resetting:'正在复位',ready:'等待动作',driving:'巡航中',episode_finished:'回合结束',closed:'已关闭',error:'异常'};
  return `<article class="card live-card"><div class="live-head"><strong>${h(workerName(w.worker))}</strong>${badge(fresh(w.updated_at)?(phases[w.phase]||w.phase):'最后记录','muted')}</div><div class="live-duration">${h(duration(w.sim_duration_sec))}</div><div class="live-details"><span>${n(w.distance_m,2)} m</span><span>${n(w.speed_mps,4)} m/s</span></div><p class="caption">第 ${n(w.episode_number)} 回合 · 标签 ${n(w.action)}<br>本段时限 ${h(duration(w.max_episode_sim_sec))}</p></article>`;
}

function renderChart(r) {
  $('chart-controls').innerHTML=Object.entries(chartSpecs).map(([key,[name]])=>`<button type="button" data-chart="${key}" aria-pressed="${key===state.chart}" class="${key===state.chart?'active':''}">${h(name)}</button>`).join('');
  const [,title,unit,multiplier]=chartSpecs[state.chart];
  const points=(r?.curves?.[state.chart] || []).map(([x,y])=>[x,y*multiplier]);
  state.chartPoints=points;
  $('chart-title').textContent=title;
  $('chart-value').textContent=points.length ? n(points[points.length-1][1],state.chart==='speed'?4:1)+' '+unit : '—';
  $('chart-note').textContent=state.chart==='duration' ? '达到时限的回合仍是正常结束，不代表此时发生碰撞。轻点曲线可查看采样点。' : '轻点曲线查看对应训练步数和数值。';
  if(!points.length){$('chart').innerHTML='<div class="empty">这个批次暂时没有该曲线数据。</div>';return;}
  const width=Math.max(280,$('chart').clientWidth || 320),height=235,left=42,right=10,top=15,bottom=30;
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(0,...ys),maxY=Math.max(...ys,0.000001)*1.08;
  const x=v=>left+(v-minX)/(maxX-minX||1)*(width-left-right),y=v=>height-bottom-(v-minY)/(maxY-minY||1)*(height-top-bottom);
  const line=points.map((p,i)=>(i?'L':'M')+x(p[0]).toFixed(2)+','+y(p[1]).toFixed(2)).join(' ');
  let grid='';
  for(let i=0;i<4;i++){const v=minY+(maxY-minY)*i/3;grid+=`<line x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}" stroke="#e4ebef"/><text class="chart-label" x="${left-7}" y="${y(v)+4}" text-anchor="end">${h(n(v,state.chart==='speed'?2:0))}</text>`;}
  for(let i=0;i<3;i++){const v=minX+(maxX-minX)*i/2;grid+=`<text class="chart-label" x="${x(v)}" y="${height-6}" text-anchor="${i===0?'start':i===2?'end':'middle'}">${h(n(v/1000,1))}k</text>`;}
  $('chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${h(title)}随训练步数的变化">${grid}<path d="${line} L${x(maxX)},${height-bottom} L${x(minX)},${height-bottom} Z" fill="#087f7410"/><path d="${line}" fill="none" stroke="#087f74" stroke-width="2.5" stroke-linejoin="round"/>${points.length===1?`<circle cx="${x(minX)}" cy="${y(ys[0])}" r="4" fill="#087f74"/>`:''}</svg>`;
  $('chart').onclick=event=>{const rect=$('chart').getBoundingClientRect();const wanted=minX+Math.max(0,Math.min(1,(event.clientX-rect.left-left)/(width-left-right)))*(maxX-minX);const p=points.reduce((best,p)=>Math.abs(p[0]-wanted)<Math.abs(best[0]-wanted)?p:best,points[0]);$('chart-note').textContent=`训练 ${n(p[0])} 步：${title} ${n(p[1],state.chart==='speed'?4:2)} ${unit}`;};
}

function renderTrends(r){
  renderChart(r);
  const last=key=>{const a=r?.curves?.[key] || [];return a.length ? a[a.length-1][1] : null;};
  const precise=v=>!valid(v)?'—':Math.abs(v)<.001 && v!==0?Number(v).toExponential(2):n(v,3);
  $('optimizer-metrics').innerHTML=metric('价值损失',precise(last('value_loss')),'观察价值估计的拟合')+metric('Approx. KL',precise(last('kl')),'观察策略更新幅度')+metric('解释方差',precise(last('explained_variance')),'价值估计指标')+metric('采样吞吐',n(r?.samples_per_sec,1),'训练样本 / 电脑秒，含评估等待');
}

function renderEpisodes(r){
  const entries=state.episodeSource==='validated' ? [...(state.data.validated_model?.episodes || [])].reverse() : (r?.episodes || []);
  const filtered=entries.filter(e=>state.reason==='all' || (state.reason==='failure' ? e.reason!=='time_limit' : e.reason===state.reason));
  $('episode-summary').textContent=`${state.episodeSource==='validated'?'已验证模型的独立验收':'选中训练批次的近期探索回合'} · 符合筛选 ${filtered.length} 段`;
  $('episode-list').innerHTML=filtered.slice(0,state.limit).map(e=>`<article class="episode"><header>${badge(reasonNames[e.reason],e.reason==='time_limit'?'':e.reason==='collision'?'danger':'warn')}<span class="caption">${h(workerName(e.worker))} · #${n(e.number)}</span></header><div class="episode-time">${e.reason==='time_limit'?'≥ ':''}${h(duration(e.sim_duration_sec))}</div><p>${e.reason==='time_limit'?'达到测试时限，尚未发生失败':'该失败发生在本回合上述时间'} · ${n(e.distance_m,2)} m</p><p>均速 ${n(e.mean_speed_mps,4)} m/s · 速度合规 ${pct(e.speed_compliance)}${e.ended_at?' · '+h(date(e.ended_at)):''}</p></article>`).join('') || '<div class="card empty">没有符合条件的记录。</div>';
  $('more-episodes').hidden=filtered.length<=state.limit;
}

function renderModel(r){
  const m=state.data.validated_model,v=m?.vehicle,p=m?.ppo;
  $('model-badge').className='badge'+(m?.verified?'':' muted'); $('model-badge').textContent=m?.verified?'已独立验收':'尚未验证';
  $('model-details').innerHTML=m ? detailList([
    ['算法 / 网络','PPO / '+(p?.POLICY_NET_ARCH||[]).join(' → ')],['训练采样',n(m.steps)+' 步'],
    ['模型输入',m.observation_mode==='privileged' ? `${m.observation_spec?.settings?.grid_rows}×${m.observation_spec?.settings?.grid_cols} 地图距离场＋3维状态` : n(v?.lidar_point_count)+' 列雷达距离'],['动作标签',`0～${n((v?.label_count || 1)-1)}，${n(Math.floor((v?.label_count || 0)/2))} 为直行`],
    ['目标前进速度',n(v?.forward_speed_mps,2)+' m/s'],[m.observation_mode==='privileged'?'雷达依赖':'雷达频率',m.observation_mode==='privileged'?'已关闭，不参与决策与奖励':n(v?.lidar_scan_hz,1)+' Hz'],
    ['已验证地图',(m.reports || []).map(r=>String(r.world || '').split('/').pop()).filter((x,i,a)=>a.indexOf(x)===i).join('、')],
    ['验证完成时间',date(m.updated_at,true)],['模型 SHA-256',m.sha256],['模型目录',m.id],
  ]) : '<div class="empty">完成独立长时验证并冻结模型后显示参数。</div>';
  const rp=r?.ppo,env=r?.environment;
  $('run-details').innerHTML=r ? detailList([
    ['批次',r.id],['观测模式',modeName(r.observation_mode)],['输入维度',r.observation_spec?.dimension || r.vehicle?.lidar_point_count],['状态',statusNames[r.status]||r.status],['地图',r.world],['并行训练车',r.num_envs],
    ['开始 / 结束',date(r.started_at)+' / '+date(r.ended_at)],['仿真控制步长',n(env?.step_sim_sec,3)+' 秒'],
    ['训练时间预算',duration(r.time_budget_sec)],['预计结束训练',date(r.training_deadline,true)],
    ['单回合时限',duration(r.episode_limit_sec)],['卡住窗口',duration(env?.stuck_window_sec)],
    ['卡住位移阈值',n(env?.stuck_min_travel_m,3)+' m'],['速度合规范围','目标 ±'+n((env?.speed_tolerance_ratio || 0)*100)+'%'],
    ['学习率',rp?.LEARNING_RATE],['折扣因子 γ',rp?.GAMMA],['熵系数',rp?.ENTROPY_COEFF],
    ['Rollout / 批次',`${r.rollout_steps || '—'} / ${rp?.BATCH_SIZE || '—'}`],['计算设备 / 种子',`${r.device} / ${r.seed}`],
  ]) : '<div class="empty">尚无训练批次。</div>';
  for(const [id,name] of [['validation-image','media/validation.png'],['learning-image','media/learning.png']]){
    const version=state.data.media?.[name];
    const url=(state.fallback?'':state.config.asset_base)+name+'?v='+encodeURIComponent(version || '');
    const container=$(id);
    if(container.dataset.version===version && container.dataset.fallback===String(state.fallback)) continue;
    container.dataset.version=version || '';container.dataset.fallback=String(state.fallback);
    container.innerHTML=version ? `<a class="image-link" href="${h(url)}" target="_blank" rel="noopener">打开原图，可双指缩放 ↗</a><img class="report-image" src="${h(url)}" loading="lazy" alt="${id==='validation-image'?'独立验收轨迹与实测速度':'模型训练曲线'}">` : '<p class="caption">暂无对应图像。</p>';
  }
}

function render(){
  const select=$('run-select'), selection=state.runId;
  select.innerHTML='<option value="latest">跟随最新训练批次</option>'+(state.data.runs || []).map(r=>`<option value="${h(r.id)}">${h(date(r.started_at))} · ${h(modeName(r.observation_mode))} · ${n(r.steps)} 步 · ${h(statusNames[r.status]||r.status)}</option>`).join('');
  if(selection!=='latest' && !state.data.runs.some(r=>r.id===selection)) state.runId='latest';
  select.value=state.runId;
  const r=run();
  renderFreshness(); renderOverview(r); renderTrends(r); renderEpisodes(r); renderModel(r);
}

async function fetchJSON(url){
  const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),12000);
  try{
    const target=new URL(url,location.href);target.searchParams.set('_',String(Math.floor(Date.now()/1000)));
    const response=await fetch(target,{cache:'no-store',signal:controller.signal,credentials:'omit'});
    if(!response.ok) throw new Error('HTTP '+response.status);
    return await response.json();
  }finally{clearTimeout(timer);}
}

async function refresh(){
  if(state.busy) return;
  state.busy=true; $('refresh').disabled=true;
  try{
    if(!state.config) state.config=await fetchJSON('config.json');
    let data, fallback=false;
    try{data=await fetchJSON(state.config.data_url);}
    catch(error){
      if(!state.data && state.config.fallback_url){data=await fetchJSON(state.config.fallback_url);fallback=true;}
      else throw error;
    }
    if(data.schema_version!==1 || !Array.isArray(data.runs)) throw new Error('快照格式不兼容');
    // Never replace a newer snapshot by an older cached response.
    if(state.data && new Date(data.generated_at)<new Date(state.data.generated_at)) throw new Error('数据源返回了较旧快照');
    state.data=data;state.error=null;state.fallback=fallback;render();
  }catch(error){
    state.error=String(error);renderFreshness();
    if(!state.data){$('hero').innerHTML='<p class="eyebrow">尚未取得进度</p><div class="hero-value">等待连接</div><p>请确认电脑端进度服务已启动，然后点击右上角刷新。</p>';$('notice').hidden=false;$('notice').textContent='无法读取进度数据。请通过 HTTP 网页地址访问，而不是直接打开本地 HTML 文件。';$('connection-badge').textContent='连接失败';$('data-age').textContent='暂无可用快照';}
  }finally{state.busy=false;$('refresh').disabled=false;}
}

$('refresh').addEventListener('click',refresh);
$('run-select').addEventListener('change',e=>{state.runId=e.target.value;state.limit=20;render();});
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{
  state.tab=button.dataset.tab;
  document.querySelectorAll('[role="tabpanel"]').forEach(p=>p.hidden=p.id!==state.tab);
  document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',String(b===button));});
  if(state.data) render();window.scrollTo({top:0,behavior:'auto'});
}));
$('chart-controls').addEventListener('click',event=>{const b=event.target.closest('[data-chart]');if(b){state.chart=b.dataset.chart;renderChart(run());}});
$('episode-source').addEventListener('click',event=>{const b=event.target.closest('[data-source]');if(b){state.episodeSource=b.dataset.source;state.limit=20;document.querySelectorAll('[data-source]').forEach(x=>x.classList.toggle('active',x===b));renderEpisodes(run());}});
$('reason-select').addEventListener('change',event=>{state.reason=event.target.value;state.limit=20;renderEpisodes(run());});
$('more-episodes').addEventListener('click',()=>{state.limit+=20;renderEpisodes(run());});
window.addEventListener('resize',()=>{if(state.data && state.tab==='trends') renderChart(run());});
document.addEventListener('visibilitychange',()=>{if(!document.hidden) refresh();});
setInterval(()=>{if(!document.hidden) renderFreshness();},5000);
(async function poll(){await refresh();setTimeout(poll,(state.config?.poll_seconds || 5)*1000);})();
