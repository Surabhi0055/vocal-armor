import puppeteer from 'puppeteer';

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.toString()));
  page.on('error', err => console.log('PAGE CRASH:', err.toString()));

  console.log('Navigating to http://localhost:5175/history');
  await page.goto('http://localhost:5175/history', { waitUntil: 'networkidle0' });
  
  await browser.close();
})();
