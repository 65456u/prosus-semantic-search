(() => {
  'use strict';
  const data = window.PROSUS_DATA;
  const main = document.getElementById('main');
  if (!data?.items?.length || !data?.queries?.length) {
    main.innerHTML = '<div class="empty-state"><h3>未读取到数据</h3><p>请先运行 prepare_data.py，再重新打开页面。</p><button class="btn" onclick="location.reload()">重新读取</button></div>';
    return;
  }
  const { items, queries, stats } = data;
  const categoryNames = {
    ALIMENTOS_PREPARADOS:'预制餐食', BEBIDAS:'饮料', MERCEARIA:'杂货食品', SAUDE_E_BEM_ESTAR:'健康与保健',
    BELEZA_ESTETICA_HIGIENE_PESSOAL:'美容与个人护理', CASA_DECORACAO:'家居', BOMBONIERE:'糖果与零食',
    OUTROS:'其他', FRIOS_LATICINIOS:'冷藏食品与乳制品', PET:'宠物用品', FLV:'水果与蔬菜', BEBE:'母婴用品',
    ELETRO_E_ELETRONICOS:'电器与电子产品', JOGOS_BRINQUEDOS:'游戏与玩具', PAPELARIA_LIVRARIA:'文具与图书',
    JARDIM_FLORICULTURA:'园艺与花卉', PROTEINAS:'肉类与蛋白食品', MODA:'服饰'
  };
  const icons = {
    overview:'<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    grid:'<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M9 9v12"/>',
    list:'<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    search:'<circle cx="10.8" cy="10.8" r="7.3"/><path d="m16 16 4.5 4.5"/>',
    layers:'<path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
    file:'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h6"/>',
    lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>',
    shop:'<path d="M3 10V5l2-2h14l2 2v5M3 10a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0M5 13v8h14v-8M9 21v-6h6v6"/>',
    package:'<path d="m12 3 9 5v9l-9 5-9-5V8l9-5ZM3 8l9 5 9-5M12 13v9M7 5.8l9 5"/>',
    image:'<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m21 16-5-5L5 21"/>',
    copy:'<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h3"/>',
    close:'<path d="m6 6 12 12M18 6 6 18"/>',
    left:'<path d="m14 6-6 6 6 6"/>', right:'<path d="m10 6 6 6-6 6"/>',
    check:'<path d="m5 12 4 4L19 6"/>', filter:'<path d="M3 5h18l-7 8v6l-4 2v-8L3 5Z"/>'
  };
  const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.package}</svg>`;
  document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); });
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const format = value => typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('en-US',{maximumFractionDigits:2}).format(value) : '未提供';
  const price = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '未提供';
  const fold = value => String(value || '').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
  const pct = count => `${(count / stats.rows * 100).toFixed(1)}%`;
  const category = item => item.itemMetadata.taxonomy?.l0 || 'OUTROS';
  const categoryLabel = key => categoryNames[key] || key;
  const searchStrings = items.map(item => fold([item.itemMetadata.name,item.itemMetadata.description,item.itemMetadata.category_name,...Object.values(item.itemMetadata.taxonomy || {}),item.itemId,item.merchantId,item._id].join(' ')));
  const byId = new Map();
  items.forEach(item => { if (!byId.has(item.itemId)) byId.set(item.itemId,item); });
  const duplicateIds = new Set(stats.duplicateIds);
  const views = ['overview','catalog','queries','schema'];
  const state = { view:views.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview', query:'',category:'',missingDescription:false,missingImage:false,duplicates:false,merchant:'',sort:'source',layout:'grid',page:1,queryText:'',showImages:false };
  try { state.showImages = localStorage.getItem('prosus-show-images') === 'true'; } catch {}
  let selectedItem = null, detailTab = 'info', selectedImage = 0, toastTimer, searchTimer;
  const pageSize = 24;
  const dialog = document.getElementById('detail-dialog');
  const mainTitles = {overview:'数据概览',catalog:'商品目录',queries:'查询集',schema:'字段说明'};

  function imageURL(path) {
    if (typeof path !== 'string' || !path.trim()) return '';
    return 'https://static.ifood-static.com.br/image/upload/t_low/pratos/' + path.split('/').map(encodeURIComponent).join('/');
  }
  function visual(item, className) {
    const images = item.itemMetadata.images || [];
    const src = state.showImages ? imageURL(images[0]) : '';
    return `<div class="${className}">${src ? `<img src="${escape(src)}" alt="${escape(item.itemMetadata.name)}" loading="lazy" referrerpolicy="no-referrer">` : icon('package')}${className === 'product-image' ? (images.length ? `<span class="image-count">${icon('image')} ${images.length}</span>` : '<span class="image-count">无图片路径</span>') : ''}</div>`;
  }
  function imageToggle() { return `<label class="image-toggle"><input type="checkbox" data-action="images" ${state.showImages?'checked':''}>加载商品图片</label>`; }
  function pageHead(eyebrow,title,subtitle,actions='') {
    return `<div class="page-head"><div><p class="eyebrow">${eyebrow}</p><h1>${title}</h1>${subtitle ? `<p class="subtitle">${subtitle}</p>` : ''}</div><div class="head-actions">${actions}</div></div>`;
  }
  function navigate(view) {
    if (!views.includes(view)) return;
    state.view = view;
    if (location.hash !== '#' + view) location.hash = view;
    render();
    window.scrollTo({top:0,behavior:'instant'});
  }
  function goCatalog(filters = {}) {
    Object.assign(state,{query:'',category:'',missingDescription:false,missingImage:false,duplicates:false,merchant:'',page:1},filters);
    navigate('catalog');
  }
  function render() {
    document.title = `${mainTitles[state.view]} · Prosus 数据浏览器`;
    document.querySelectorAll('.navigation a').forEach(el => {
      const active = el.dataset.view === state.view;
      el.classList.toggle('active',active);
      if (active) el.setAttribute('aria-current','page'); else el.removeAttribute('aria-current');
    });
    if (state.view === 'overview') renderOverview();
    if (state.view === 'catalog') renderCatalog();
    if (state.view === 'queries') renderQueries();
    if (state.view === 'schema') renderSchema();
  }
  function renderOverview() {
    const colors = ['#176b4a','#72a78b','#a4cbaa','#adc7e6','#aaa1cc','#e0e7e4'];
    const segments = [...stats.categories.slice(0,5),{key:'remaining',count:stats.categories.slice(5).reduce((n,c)=>n+c.count,0)}];
    let total = 0;
    const gradient = segments.map((segment,index) => { const start = total / stats.rows * 100; total += segment.count; return `${colors[index]} ${start}% ${total / stats.rows * 100}%`; }).join(',');
    const previews = [{index:1,zh:'柴火烤制的薄底披萨'},{index:24,zh:'米纳斯地区家常菜'},{index:29,zh:'素食蛋白碗'},{index:77,zh:'训练后的蛋白餐'},{index:87,zh:'品牌与产品的明确检索'}];
    const samplePatterns = [/^McChicken Duplo$/i,/pizza/i,/^Macarrão Pena com Ovos Adria/i,/^Desinfetante Uso Geral Lavanda Lysoform/i];
    const samples = samplePatterns.map(pattern => items.find(item=>pattern.test(item.itemMetadata.name))).filter(Boolean);
    main.innerHTML = pageHead('DATA OVERVIEW','数据概览','餐饮、商超与非食品商品，共用一个目录。',`<button class="btn btn-primary" data-action="catalog">${icon('grid')}浏览商品</button>`) +
      `<div class="stats-grid">
        <div class="stat-card featured"><div class="stat-label">商品记录${icon('package')}</div><div class="stat-value">${format(stats.rows)}</div><div class="stat-caption">${format(stats.uniqueItems)} 个不同的商品 ID</div></div>
        <div class="stat-card"><div class="stat-label">商家${icon('shop')}</div><div class="stat-value">${format(stats.merchants)}</div><div class="stat-caption">按 merchantId 去重</div></div>
        <div class="stat-card"><div class="stat-label">评估查询${icon('search')}</div><div class="stat-value">${format(stats.queries)}</div><div class="stat-caption">葡萄牙语 · 未提供相关性标签</div></div>
        <div class="stat-card"><div class="stat-label">带图片路径${icon('image')}</div><div class="stat-value">${(stats.images/stats.rows*100).toFixed(0)}<small>%</small></div><div class="stat-caption">${format(stats.images)} / ${format(stats.rows)} 条记录</div></div>
      </div>
      <div class="overview-grid">
        <section class="panel"><div class="panel-head"><h2>商品构成</h2><span class="muted">一级分类 · 原始行数</span></div>
          <div class="composition"><div class="donut" style="background:conic-gradient(${gradient})" role="img" aria-label="商品一级分类分布，详细数值见右侧列表"><div class="donut-core"><div class="donut-number">${format(stats.rows)}</div><div class="donut-label">条商品记录</div></div></div>
          <div class="legend">${segments.map((s,i)=>`<button class="legend-row" data-category="${s.key==='remaining'?'':escape(s.key)}"><span class="legend-swatch" style="background:${colors[i]}"></span><span>${s.key==='remaining'?'其余 13 个类别':categoryLabel(s.key)}</span><span class="legend-value">${format(s.count)} <span>· ${pct(s.count)}</span></span></button>`).join('')}</div></div>
          <p class="panel-note">预制餐食占 <strong>${pct(stats.categories.find(c=>c.key==='ALIMENTOS_PREPARADOS')?.count||0)}</strong>。目录还包含商超、健康与保健、家居等商品。</p>
        </section>
        <section class="panel"><div class="panel-head"><h2>查询预览</h2><button class="text-btn" data-action="queries">查看全部 ${stats.queries} 条</button></div>
          <div class="query-preview">${previews.map(p=>`<button class="query-preview-row" data-query="${p.index}"><span class="query-number">${String(p.index+1).padStart(3,'0')}</span><div><div class="query-preview-text" lang="pt-BR">${escape(queries[p.index])}</div><div class="query-translation">${p.zh}</div></div></button>`).join('')}</div>
        </section>
      </div>
      <section class="panel"><div class="panel-head"><h2>字段覆盖情况</h2><button class="text-btn" data-action="schema">查看字段说明</button></div>
        <div class="coverage-grid">${[{label:'有商品描述',count:stats.descriptions,missing:stats.rows-stats.descriptions},{label:'有图片路径',count:stats.images,missing:stats.rows-stats.images},{label:'有搜索记录',count:stats.search,missing:stats.rows-stats.search},{label:'有共同购买记录',count:stats.coPurchase,missing:stats.rows-stats.coPurchase}].map(c=>`<div><div class="coverage-label"><span>${c.label}</span><strong>${pct(c.count)}</strong></div><div class="progress"><span style="width:${c.count/stats.rows*100}%"></span></div><div class="coverage-caption">${format(c.count)} 条有值 · ${format(c.missing)} 条为空</div></div>`).join('')}</div>
      </section>
      <div class="record-strip"><h2>商品样本</h2>${imageToggle()}</div>
      <div class="sample-grid">${samples.map(item=>`<button class="sample-card" data-item="${item.row}">${visual(item,'sample-visual')}<div class="sample-body"><div class="sample-title" lang="pt-BR">${escape(item.itemMetadata.name)}</div><div class="sample-meta"><span>${escape(categoryLabel(category(item)))}</span><span>价格 ${price(item.itemMetadata.price)}</span></div></div></button>`).join('')}</div>`;
  }
  function getFiltered() {
    const terms = fold(state.query).trim().split(/\s+/).filter(Boolean);
    let filtered = items.filter((item,index)=> (!state.category || category(item)===state.category) && (!state.merchant || item.merchantId===state.merchant) && (!state.missingDescription || !item.itemMetadata.description.trim()) && (!state.missingImage || !item.itemMetadata.images.length) && (!state.duplicates || duplicateIds.has(item.itemId)) && terms.every(term=>searchStrings[index].includes(term)));
    if (state.sort==='name') filtered.sort((a,b)=>a.itemMetadata.name.localeCompare(b.itemMetadata.name,'pt-BR'));
    if (state.sort==='priceAsc') filtered.sort((a,b)=>a.itemMetadata.price-b.itemMetadata.price);
    if (state.sort==='priceDesc') filtered.sort((a,b)=>b.itemMetadata.price-a.itemMetadata.price);
    if (state.sort==='orders') filtered.sort((a,b)=>(b.itemProfile.metrics.total_orders||0)-(a.itemProfile.metrics.total_orders||0));
    return filtered;
  }
  function renderCatalog() {
    main.innerHTML = pageHead('THE ITEM CATALOG','商品目录','搜索名称、描述、分类或 ID；保留全部原始记录。',imageToggle()) +
      `<div class="toolbar"><div class="filter-row"><label class="search-box">${icon('search')}<input id="catalog-search" aria-label="搜索商品" type="search" placeholder="搜索商品，例如 pizza、frango、Lysoform…" value="${escape(state.query)}"></label>
        <select id="category-filter" aria-label="一级分类"><option value="">全部一级分类</option>${stats.categories.map(c=>`<option value="${escape(c.key)}" ${state.category===c.key?'selected':''}>${escape(categoryLabel(c.key))} (${format(c.count)})</option>`).join('')}</select>
        <select id="sort-filter" aria-label="排序"><option value="source">原始记录顺序</option><option value="name">名称 A–Z</option><option value="priceAsc">价格从低到高</option><option value="priceDesc">价格从高到低</option><option value="orders">订单数量从高到低</option></select></div>
        <div class="filter-chips"><button class="chip ${state.category==='ALIMENTOS_PREPARADOS'?'active':''}" data-action="prepared">预制餐食 ${format(stats.categories.find(c=>c.key==='ALIMENTOS_PREPARADOS')?.count||0)}</button><button class="chip ${state.missingDescription?'active':''}" data-filter="missingDescription">缺少描述 ${format(stats.rows-stats.descriptions)}</button><button class="chip ${state.missingImage?'active':''}" data-filter="missingImage">无图片 ${format(stats.rows-stats.images)}</button><button class="chip ${state.duplicates?'active':''}" data-filter="duplicates">重复 ID 的记录 ${items.filter(i=>duplicateIds.has(i.itemId)).length}</button>${state.merchant?`<button class="chip active" data-action="clear-merchant">同商家 ${escape(state.merchant.slice(0,8))}… ${icon('close')}</button>`:''}<button class="reset-btn" data-action="reset">清除筛选</button></div>
      </div><div id="catalog-results" aria-live="polite"></div>`;
    document.getElementById('sort-filter').value=state.sort;
    document.getElementById('catalog-search').addEventListener('input',event=>{state.query=event.target.value;state.page=1;clearTimeout(searchTimer);searchTimer=setTimeout(renderCatalogResults,100);});
    document.getElementById('category-filter').addEventListener('change',event=>{state.category=event.target.value;state.page=1;renderCatalog();});
    document.getElementById('sort-filter').addEventListener('change',event=>{state.sort=event.target.value;state.page=1;renderCatalogResults();});
    renderCatalogResults();
  }
  function renderCatalogResults() {
    if (state.view!=='catalog') return;
    const results = getFiltered(), pages = Math.max(1,Math.ceil(results.length/pageSize));
    state.page=Math.min(Math.max(1,state.page),pages);
    const start=(state.page-1)*pageSize, current=results.slice(start,start+pageSize);
    const content = !results.length ? `<div class="empty-state">${icon('search')}<h3>没有匹配的商品</h3><p>尝试较短的关键词，或清除其他筛选条件。</p><button class="btn" data-action="reset">清除筛选</button></div>` : state.layout==='grid' ?
      `<div class="product-grid">${current.map(item=>{const m=item.itemMetadata;return `<button class="product-card" data-item="${item.row}">${visual(item,'product-image')}<div class="product-body"><div class="product-category">${escape(categoryLabel(category(item)))}</div><div class="product-name" lang="pt-BR">${escape(m.name)}</div><div class="product-description" lang="pt-BR">${m.description ? escape(m.description) : '<span class="muted">未提供商品描述</span>'}</div><div class="product-bottom"><span class="product-price"><span class="price-label">价格</span>${price(m.price)}</span><span class="orders-label">${format(item.itemProfile.metrics.total_orders)} 订单</span></div></div></button>`;}).join('')}</div>` :
      `<div class="product-table-wrapper"><table class="data-table"><thead><tr><th>商品</th><th>一级分类</th><th class="right">价格</th><th class="right">订单数量</th><th>记录</th></tr></thead><tbody>${current.map(item=>{const m=item.itemMetadata,src=state.showImages?imageURL(m.images?.[0]):'';return `<tr data-item="${item.row}"><td><div class="table-product">${src?`<img class="table-image" src="${escape(src)}" alt="" loading="lazy" referrerpolicy="no-referrer">`:`<div class="table-image table-placeholder">${icon('package')}</div>`}<div class="table-name" lang="pt-BR">${escape(m.name)}<small>${escape(m.description||'未提供商品描述')}</small></div></div></td><td><span class="badge">${escape(categoryLabel(category(item)))}</span></td><td class="right">${price(m.price)}</td><td class="right">${format(item.itemProfile.metrics.total_orders)}</td><td><button class="text-btn" data-item="${item.row}">#${item.row}</button></td></tr>`;}).join('')}</tbody></table></div>`;
    document.getElementById('catalog-results').innerHTML=`<div class="results-bar"><span>找到 <strong id="result-count">${format(results.length)}</strong> 条记录 <span> / ${format(stats.rows)}</span></span><div class="view-toggle" aria-label="展示方式"><button aria-label="卡片视图" aria-pressed="${state.layout==='grid'}" class="${state.layout==='grid'?'active':''}" data-layout="grid">${icon('overview')}</button><button aria-label="表格视图" aria-pressed="${state.layout==='table'}" class="${state.layout==='table'?'active':''}" data-layout="table">${icon('list')}</button></div></div>${content}${results.length?pagination(results.length,pages,start,current.length):''}`;
  }
  function pagination(count,pages,start,length) {
    const numbers=[...new Set([1,...[state.page-1,state.page,state.page+1].filter(p=>p>1&&p<pages),pages])].sort((a,b)=>a-b);
    let previous=0;
    const buttons=numbers.map(number=>{const gap=previous&&number-previous>1?'<span class="page-ellipsis">…</span>':'';previous=number;return `${gap}<button class="page-btn ${number===state.page?'active':''}" data-page="${number}" ${number===state.page?'aria-current="page"':''}>${number}</button>`;}).join('');
    return `<div class="pagination"><span>显示 ${format(start+1)}–${format(start+length)} 条，共 ${format(count)} 条</span><div class="page-controls"><button class="page-btn" data-page="${state.page-1}" aria-label="上一页" ${state.page===1?'disabled':''}>${icon('left')}</button>${buttons}<button class="page-btn" data-page="${state.page+1}" aria-label="下一页" ${state.page===pages?'disabled':''}>${icon('right')}</button></div></div>`;
  }
  function renderQueries() {
    main.innerHTML=pageHead('THE SEARCH INTENTS','查询集','100 条评估查询，保留原始顺序与葡语文本。')+`<div class="search-box"><span>${icon('search')}</span><input id="query-search" type="search" aria-label="搜索查询" placeholder="筛选查询，例如 pizza、café、proteína…" value="${escape(state.queryText)}"></div><p class="query-hint">“按原词筛选”用于浏览目录，不代表语义检索或相关性排名。</p><div id="query-results" aria-live="polite"></div>`;
    document.getElementById('query-search').addEventListener('input',event=>{state.queryText=event.target.value;renderQueryResults();});
    renderQueryResults();
  }
  function renderQueryResults() {
    const matching=queries.map((text,index)=>({text,index})).filter(q=>fold(q.text).includes(fold(state.queryText).trim()));
    document.getElementById('query-results').innerHTML=`<div class="results-bar">显示 <strong id="query-count">${matching.length}</strong> / ${queries.length} 条查询</div>`+(matching.length?`<div class="query-grid">${matching.map(q=>`<article class="query-card"><div class="query-card-top"><span class="query-number">QUERY ${String(q.index+1).padStart(3,'0')}</span><span class="badge">葡萄牙语</span></div><div class="query-text" lang="pt-BR">${escape(q.text)}</div><div class="query-card-actions"><button class="text-btn" data-query="${q.index}">按原词筛选商品</button><button class="icon-btn" data-copy-query="${q.index}" aria-label="复制查询 ${q.index+1}">${icon('copy')}</button></div></article>`).join('')}</div>`:`<div class="empty-state"><h3>没有匹配的查询</h3><p>尝试其他关键词。</p></div>`);
  }
  function schemaTable(title,sub,fields) {
    return `<section class="panel schema-panel"><div class="panel-head"><h2>${title}</h2><span class="muted">${sub}</span></div><table class="data-table"><thead><tr><th>字段</th><th>类型</th><th>内容</th></tr></thead><tbody>${fields.map(f=>`<tr><td>${f[0]}</td><td>${f[1]}</td><td>${f[2]}</td></tr>`).join('')}</tbody></table></section>`;
  }
  function renderSchema() {
    main.innerHTML=pageHead('THE DATA STRUCTURE','字段说明','两个 CSV 文件。商品表的两个字段包含嵌套 JSON。')+
      `<div class="schema-summary"><span class="badge badge-green">商品表 · ${format(stats.rows)} 行 / 5 列</span><span class="badge">查询表 · ${queries.length} 行 / 1 列</span><span class="badge">${stats.duplicateRows} 条额外重复商品记录</span></div>`+
      schemaTable('商品记录','5k_items_curated.csv',[['_id','string','记录 ID；所有 5,000 行各不相同。'],['itemId','string',`${format(stats.uniqueItems)} 个不同商品 ID；一个商品 ID 出现了 4 次。`],['merchantId','string',`${format(stats.merchants)} 个商家 ID；未附带商家详情表。`],['itemMetadata','object','名称、描述、分类、价格、图片与商品标记。'],['itemProfile','object','搜索关联记录与商品行为统计。']])+
      schemaTable('商品信息','itemMetadata',[['name','string','商品名称，保留葡语原文。'],['description','string',`${format(stats.rows-stats.descriptions)} 行为空；部分非空描述只有包装规格。`],['category_name','string','类目名称，如三明治、清洁用品。'],['taxonomy','object','l0 / l1 / l2 三级商品分类。'],['price','number','原始价格值；数据说明未明确币种。'],['images','string[]',`${format(stats.images)} 行有图片路径；通过提供的 iFood 地址加载。`],['tags','object[]','商品标签；每条含 key 和 value，例如份量标签。'],['vegan','boolean',`${stats.flags.vegan} 行标记为 true；标注完整性未说明。`],['lacFree','boolean',`${stats.flags.lacFree} 行标记为 true；表示无乳糖标记。`],['organic','boolean',`${stats.flags.organic} 行标记为 true；表示有机标记。`]])+
      schemaTable('行为统计','itemProfile.metrics',[['total_orders','number','订单数量；未提供统计时间窗口。'],['avgBasketPrice','number','平均购物篮价格原始值。'],['conversionRate','number','转化相关统计；值可大于 1，不能直接按 0–1 概率解释。'],['reorderRate','number?', '复购相关统计；7 行没有该字段。'],['voucherUsage','number','优惠券使用相关统计，值可大于 1。'],['deliveryTime','number','配送时间原始值；时间单位未说明。'],['preparationTime','number','准备时间原始值；时间单位未说明。'],['orderingRate','object','day：工作日／周末；shift：早餐、凌晨、晚餐、午餐、加餐。'],['coPurchaseItems','object[]',`${format(stats.coPurchase)} 行有记录。含商品 ID、商家 ID、名称、描述、价格，部分商品不在当前目录。`],['customizations','object[]',`${format(stats.customizations)} 行有记录。含 option_id、name、avg_quantity 和 avg_unit_value。`]])+
      schemaTable('搜索关联记录','itemProfile.search[]',[['term','string','关联的搜索词；不一定描述商品本身。'],['count','number','该关联记录的计数。'],['method','string?','来源方式：term、suggestion、historic 等；少量记录未提供。']])+
      schemaTable('评估查询','queries.csv',[['search_term_pt','string','100 条不重复的葡语查询；没有提供对应的正确商品列表。']])+
      '<div class="schema-note">数据没有提供查询与商品的相关性标签、标准答案、训练／验证划分、用户级事件记录或完整的商家资料。搜索关联记录和共同购买记录可以提供线索，但不能直接当作相关性真值。</div>';
  }
  function openItem(row) {
    const item=items[Number(row)-1];
    if (!item) return;
    selectedItem=item;detailTab='info';selectedImage=0;
    renderDetail();
    if (!dialog.open) dialog.showModal();
    document.body.style.overflow='hidden';
    dialog.scrollTop=0;
  }
  function renderDetail() {
    const item=selectedItem;if(!item)return;
    const m=item.itemMetadata;
    document.getElementById('detail-content').innerHTML=`<div class="detail-top"><div class="detail-header"><div class="detail-kicker"><span>商品记录 #${item.row} ${duplicateIds.has(item.itemId)?'· 重复商品 ID':''}</span><button class="icon-btn" data-action="close-detail" aria-label="关闭商品详情">${icon('close')}</button></div><h2 id="detail-title" class="detail-title" lang="pt-BR">${escape(m.name)}</h2><p class="detail-subtitle">${escape(categoryLabel(category(item)))} <span>·</span> ${escape(m.category_name)}</p></div><div class="detail-tabs" role="tablist" aria-label="商品详情"><button class="detail-tab ${detailTab==='info'?'active':''}" data-tab="info" role="tab" aria-controls="detail-body" aria-selected="${detailTab==='info'}">商品信息</button><button class="detail-tab ${detailTab==='profile'?'active':''}" data-tab="profile" role="tab" aria-controls="detail-body" aria-selected="${detailTab==='profile'}">行为画像</button><button class="detail-tab ${detailTab==='raw'?'active':''}" data-tab="raw" role="tab" aria-controls="detail-body" aria-selected="${detailTab==='raw'}">原始 JSON</button></div></div><div id="detail-body" class="detail-body" role="tabpanel">${detailTab==='info'?infoContent(item):detailTab==='profile'?profileContent(item):rawContent(item)}</div>`;
  }
  function infoContent(item) {
    const m=item.itemMetadata,images=m.images||[],src=state.showImages?imageURL(images[selectedImage]):'';
    return `<div class="detail-visual">${src?`<img src="${escape(src)}" alt="${escape(m.name)}" referrerpolicy="no-referrer">`:icon('package')}</div>${state.showImages&&images.length>1?`<div class="image-thumbnails">${images.map((path,i)=>`<button data-image="${i}" class="${i===selectedImage?'active':''}" aria-label="查看第 ${i+1} 张图片"><img src="${escape(imageURL(path))}" alt="" loading="lazy" referrerpolicy="no-referrer"></button>`).join('')}</div>`:''}<p class="detail-note">${!images.length?'这条记录没有图片路径。':!state.showImages?`提供 ${images.length} 个图片路径。${imageToggle()}`:`${images.length} 个图片路径 · 图片来自提供的 iFood 地址`}</p>
      <div class="detail-properties"><div class="detail-property"><div class="property-label">商品价格 · 原始值</div><div class="property-value">${price(m.price)}</div></div><div class="detail-property"><div class="property-label">订单数量</div><div class="property-value">${format(item.itemProfile.metrics.total_orders)}</div></div></div>
      <section class="detail-section"><h3>商品描述</h3><p class="detail-description" lang="pt-BR">${escape(m.description||'未提供商品描述。')}</p></section>
      <section class="detail-section"><h3>三级分类</h3><div class="taxonomy-steps">${Object.entries(m.taxonomy||{}).map(([key,value])=>`<span>${escape(key)} · ${escape(value)}</span>`).join('')}</div></section>
      <section class="detail-section"><h3>商品标记</h3><div class="flag-row">${[['vegan','纯素'],['lacFree','无乳糖'],['organic','有机']].map(([key,label])=>`<span class="badge ${m[key]?'badge-green':''}">${label} · ${String(m[key])}</span>`).join('')}</div><p class="detail-note">显示原始布尔标记；数据未说明标注完整性。</p></section>
      <section class="detail-section"><h3>商品标签</h3>${m.tags?.length?`<div class="flag-row">${m.tags.map(t=>`<span class="badge">${escape(t.key)} · ${escape(Array.isArray(t.value)?t.value.join(', '):t.value)}</span>`).join('')}</div>`:'<p class="muted">未提供标签。</p>'}</section>
      <section class="detail-section"><h3>记录标识</h3>${[['_id',item._id],['itemId',item.itemId],['merchantId',item.merchantId]].map(([key,value])=>`<div class="definition-row"><span class="definition-label">${key}</span><span class="definition-value mono">${escape(value)}</span></div>`).join('')}<div style="margin-top:16px"><button class="text-btn" data-merchant="${escape(item.merchantId)}">查看该商家的商品</button></div></section>`;
  }
  function profileContent(item) {
    const metrics=item.itemProfile.metrics,search=item.itemProfile.search||[];
    const fields=[['total_orders','订单数量'],['avgBasketPrice','平均购物篮价格'],['conversionRate','转化统计'],['reorderRate','复购统计'],['voucherUsage','优惠券使用'],['deliveryTime','配送时间'],['preparationTime','准备时间']];
    const coPurchase=metrics.coPurchaseItems||[],customizations=metrics.customizations||[];
    const ordering=metrics.orderingRate||{};
    return `<div class="detail-stats-grid">${fields.map(([key,label])=>`<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${format(metrics[key])}</div><div class="metric-key">${key}</div></div>`).join('')}</div><p class="detail-note">均显示原始值。统计窗口、时间单位，以及部分“率”的分母未说明。</p>
      <section class="detail-section"><h3>搜索关联记录 <span class="badge">${search.length}</span></h3>${search.length?detailTable(['搜索词','次数','来源方式'],search.map(s=>[escape(s.term),format(s.count),escape(s.method||'未提供')])):'<p class="detail-description">没有搜索关联记录。</p>'}<p class="detail-note">关联词未必描述商品本身，不是评估查询的标准答案。</p></section>
      <section class="detail-section"><h3>下单时间分布</h3>${detailTable(['分组','类别','原始值'],Object.entries(ordering).flatMap(([group,values])=>Object.entries(values).map(([key,value])=>[escape(group),escape(({weekday:'工作日',weekend:'周末',breakfast:'早餐',dawn:'凌晨',dinner:'晚餐',lunch:'午餐',snack:'加餐'})[key]||key),format(value)])))}</section>
      <section class="detail-section"><h3>共同购买商品 <span class="badge">${coPurchase.length}</span></h3>${coPurchase.length?`<div class="copurchase-list">${coPurchase.map(p=>`<div class="copurchase-item"><div><span lang="pt-BR">${escape(p.item_name||p.item_id||'未提供名称')}</span><small>价格 ${price(p.gross_price)}${p.item_description?' · '+escape(p.item_description):''}</small></div>${byId.has(p.item_id)?`<button data-item="${byId.get(p.item_id).row}">查看商品</button>`:'<span class="badge">不在当前目录</span>'}</div>`).join('')}</div>`:'<p class="detail-description">没有共同购买记录。</p>'}</section>
      <section class="detail-section"><h3>定制选项 <span class="badge">${customizations.length}</span></h3>${customizations.length?detailTable(['选项','平均数量','平均单价'],customizations.map(c=>[escape(c.name),format(c.avg_quantity),price(c.avg_unit_value)])):'<p class="detail-description">没有定制选项记录。</p>'}</section>`;
  }
  function detailTable(headers,rows) {return `<div class="detail-table-wrapper"><table class="data-table"><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(value=>`<td>${value}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
  function rawContent(item) {
    const {row,...original}=item;
    return `<p class="detail-note" style="margin:0 0 14px">原始 CSV 的两个 JSON 字符串已解析为对象，其余字段原样保留。</p><pre class="detail-code">${escape(JSON.stringify(original,null,2))}</pre>`;
  }
  function closeDetail() {dialog.close();document.body.style.overflow='';}
  function toast(message) {const el=document.getElementById('toast');el.textContent=message;el.classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('visible'),2400);}
  async function copyQuery(index) {
    const text=queries[index];if(typeof text!=='string')return;
    try {
      if(navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else {const input=document.createElement('textarea');input.value=text;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();const copied=document.execCommand('copy');input.remove();if(!copied)throw new Error('Clipboard unavailable');}
      toast('查询已复制');
    } catch {toast('无法自动复制，请选中原文复制。');}
  }
  function handleClick(event) {
    const navigationLink=event.target.closest('.navigation a[data-view],.brand');
    if(navigationLink){event.preventDefault();navigate(navigationLink.dataset.view||'overview');return;}
    const target=event.target.closest('button,[data-item]');if(!target)return;
    if(target.dataset.copyQuery!==undefined){copyQuery(Number(target.dataset.copyQuery));return;}
    if(target.dataset.query!==undefined){const index=Number(target.dataset.query);if(queries[index]!==undefined)goCatalog({query:queries[index]});return;}
    if(target.dataset.category!==undefined){goCatalog({category:target.dataset.category});return;}
    if(target.dataset.item!==undefined){openItem(target.dataset.item);return;}
    if(target.dataset.merchant!==undefined){closeDetail();goCatalog({merchant:target.dataset.merchant});return;}
    if(target.dataset.tab){detailTab=target.dataset.tab;renderDetail();dialog.querySelector(`[data-tab="${detailTab}"]`)?.focus();return;}
    if(target.dataset.image!==undefined){selectedImage=Number(target.dataset.image);renderDetail();return;}
    if(target.dataset.filter){state[target.dataset.filter]=!state[target.dataset.filter];state.page=1;renderCatalog();return;}
    if(target.dataset.layout){state.layout=target.dataset.layout;renderCatalogResults();return;}
    if(target.dataset.page!==undefined){state.page=Number(target.dataset.page);renderCatalogResults();document.getElementById('catalog-results')?.scrollIntoView({block:'start',behavior:'instant'});return;}
    const action=target.dataset.action;
    if(action==='catalog')goCatalog();
    if(action==='queries'||action==='schema')navigate(action);
    if(action==='reset')goCatalog();
    if(action==='prepared'){state.category=state.category==='ALIMENTOS_PREPARADOS'?'':'ALIMENTOS_PREPARADOS';state.page=1;renderCatalog();}
    if(action==='clear-merchant'){state.merchant='';state.page=1;renderCatalog();}
    if(action==='close-detail')closeDetail();
  }
  document.addEventListener('click',handleClick);
  document.addEventListener('change',event=>{
    if(event.target.dataset.action==='images'){
      state.showImages=event.target.checked;
      try{localStorage.setItem('prosus-show-images',String(state.showImages));}catch{}
      render();if(dialog.open)renderDetail();
    }
  });
  document.addEventListener('error',event=>{
    if(event.target.tagName==='IMG'){
      const placeholder=document.createElement('span');placeholder.className='missing-image';placeholder.textContent='图片暂不可用';
      event.target.replaceWith(placeholder);
    }
  },true);
  dialog.addEventListener('close',()=>{document.body.style.overflow='';});
  dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)closeDetail();}});
  dialog.addEventListener('keydown',event=>{
    if(event.target.getAttribute('role')!=='tab'||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const tabs=['info','profile','raw'];let index=tabs.indexOf(detailTab);
    index=event.key==='Home'?0:event.key==='End'?2:(index+(event.key==='ArrowRight'?1:2))%3;
    detailTab=tabs[index];renderDetail();dialog.querySelector(`[data-tab="${detailTab}"]`)?.focus();
  });
  window.addEventListener('hashchange',()=>{const view=location.hash.slice(1);if(views.includes(view)&&view!==state.view){state.view=view;render();window.scrollTo({top:0,behavior:'instant'});}});
  render();

  // Expose the same catalogue filtering action when the browser supports WebMCP.
  const modelContext=document.modelContext;
  if(modelContext?.registerTool){
    const lifecycle=new AbortController();
    try{Promise.resolve(modelContext.registerTool({name:'filter_catalog',title:'筛选商品目录',description:'Navigate to the local catalogue and filter item records by text and optional level-zero taxonomy. Text filtering is keyword matching, not semantic retrieval.',inputSchema:{type:'object',properties:{query:{type:'string'},category:{type:'string'}},additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:true},execute(input){if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).some(k=>!['query','category'].includes(k))||(input.query!==undefined&&typeof input.query!=='string')||(input.category!==undefined&&typeof input.category!=='string')||(input.category&&!stats.categories.some(c=>c.key===input.category)))throw new Error('Invalid catalogue filter');goCatalog({query:input.query||'',category:input.category||''});const matches=getFiltered();return {count:matches.length,items:matches.slice(0,20).map(i=>({row:i.row,itemId:i.itemId,name:i.itemMetadata.name}))};}},{signal:lifecycle.signal})).catch(()=>{});}catch{}
    window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  }
})();
