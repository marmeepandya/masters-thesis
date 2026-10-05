const pptxgen = require('pptxgenjs');
const html2pptx = require('./html2pptx');
const path = require('path');

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Marmee Pandya';
  pptx.title = 'Can PDL replace the GOI for an open-source paper?';

  const dir = path.join(__dirname, 'html');

  await html2pptx(path.join(dir, '01_title.html'), pptx);
  await html2pptx(path.join(dir, '02_why.html'), pptx);
  await html2pptx(path.join(dir, '03_method.html'), pptx);

  const { slide: slide4, placeholders } = await html2pptx(path.join(dir, '04_schema.html'), pptx);
  const tableArea = placeholders.find(p => p.id === 'table-area');

  const hdr = { fill: { color: '1C2833' }, color: 'FFFFFF', bold: true, fontSize: 9.5 };
  const normal = { fontSize: 9, color: '2E4053' };
  const critical = { fontSize: 9, color: '7B241C', fill: { color: 'FDEDEC' }, bold: true };

  const rows = [
    [{ text: 'GOI field', options: hdr }, { text: 'PDL field', options: hdr }, { text: 'Note', options: hdr }],
    ['domain', 'website', 'join key for overlap'],
    ['name', 'name', 'direct match'],
    ['country', 'country', 'direct match'],
    ['state', 'region', 'approximate match'],
    ['district / municipality', 'locality', 'GOI has two finer levels'],
    ['organization_size', 'size', 'bucket boundaries differ'],
    ['nace_code', 'industry', 'different taxonomies entirely'],
    ['organization_type', '—', 'no PDL equivalent'],
    ['—', 'founded', 'no GOI equivalent'],
    ['—', 'linkedin_url', 'no GOI equivalent'],
    [{ text: 'summary', options: critical }, { text: '—', options: critical }, { text: 'MISSING: the field this thesis’ retrieval method depends on most', options: critical }],
    [{ text: 'summary_keywords', options: critical }, { text: '—', options: critical }, { text: 'MISSING: derived from summary, same gap', options: critical }],
  ].map((row, i) => i === 0 ? row : row.map(c => typeof c === 'string' ? { text: c, options: normal } : c));

  slide4.addTable(rows, {
    ...tableArea,
    colW: [1.9, 1.5, 3.1],
    border: { pt: 0.5, color: 'D5DBDE' },
    valign: 'middle',
    autoPage: false,
  });

  await html2pptx(path.join(dir, '05_headline.html'), pptx);
  await html2pptx(path.join(dir, '06_geography.html'), pptx);
  await html2pptx(path.join(dir, '07_holds_up.html'), pptx);
  await html2pptx(path.join(dir, '08_missing_piece.html'), pptx);
  await html2pptx(path.join(dir, '09_verdict.html'), pptx);
  await html2pptx(path.join(dir, '10_discussion.html'), pptx);

  await pptx.writeFile({ fileName: path.join(__dirname, '..', 'PDL_vs_GOI_comparison.pptx') });
  console.log('Presentation created successfully!');
}

build().catch(err => { console.error(err); process.exit(1); });
