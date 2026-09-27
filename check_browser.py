"""Verify report data, controls, anatomical images, and responsive layout."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((ROOT/'index.html').as_uri())
    assert page.title()=='Lincoln traffic control experiment'
    font=page.locator('body').evaluate('(e)=>getComputedStyle(e).fontFamily')
    assert 'Times New Roman' in font
    assert page.locator('img').evaluate_all('(els)=>els.every(e=>e.complete&&e.naturalWidth>0)')
    page.get_by_text('Individual neuron samples and full connectivity matrix',exact=True).click()
    assert page.get_by_alt_text('Ten labeled neuron skeletons with MaleCNS body IDs and 20 micrometer scale bars').is_visible()
    page.screenshot(path=str(ROOT/'results/report-desktop.png'),full_page=True)
    page.get_by_text('Individual neuron samples and full connectivity matrix',exact=True).click()
    baseline=page.locator('#delay').inner_text()
    page.get_by_label('Controller',exact=True).select_option('fixed')
    fixed=page.locator('#delay').inner_text()
    assert float(fixed)>float(baseline)
    page.get_by_label('Scenario',exact=True).select_option('restriction')
    restriction=page.locator('#delay').inner_text()
    assert float(restriction)>float(fixed)
    page.get_by_role('button',name='Play',exact=True).click()
    page.wait_for_function('Number(document.getElementById("time").value)>0')
    page.get_by_role('button',name='Pause',exact=True).click()
    page.get_by_role('button',name='Reset',exact=True).click()
    assert page.get_by_label('Simulation time',exact=True).input_value()=='0'
    page.get_by_label('Simulation time',exact=True).fill('359')
    assert page.locator('#queued').inner_text()=='0'
    assert page.locator('#arrived').inner_text()==page.locator('#departed').inner_text()
    page.get_by_label('Scenario',exact=True).select_option('standard')
    page.get_by_label('Controller',exact=True).select_option('connectome')
    page.get_by_label('Simulation time',exact=True).fill('75')
    page.locator('#simulation').screenshot(path=str(ROOT/'results/simulation-desktop.png'))
    page.get_by_text('Neural ablation results',exact=True).click()
    assert page.locator('#ablations tr').count()==5
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
    page.screenshot(path=str(ROOT/'results/report-mobile.png'),full_page=True)
    assert not errors,errors
    result=dict(font=font,all_images_loaded=True,standard_circuit_seconds=baseline,standard_fixed_seconds=fixed,
                restriction_fixed_seconds=restriction,play_pause_reset=True,end_conservation=True,
                ablation_rows=5,mobile_overflow=False,page_errors=errors)
    (ROOT/'results/browser-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));browser.close()
