const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function run() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
  const dir = path.join(__dirname, 'html');
  const outDir = path.join(__dirname, 'previews');
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.html')).sort();
  for (const f of files) {
    await page.goto('file://' + path.join(dir, f));
    await page.screenshot({ path: path.join(outDir, f.replace('.html', '.png')) });
    console.log('Captured', f);
  }
  await browser.close();
}

run().catch(err => { console.error(err); process.exit(1); });
