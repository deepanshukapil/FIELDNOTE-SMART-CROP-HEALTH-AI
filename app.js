const cropForm = document.querySelector('#crop-form');
const cropResult = document.querySelector('#crop-result');
const targetCrop = document.querySelector('#target-crop');
const leafInput = document.querySelector('#leaf-file');
const leafButton = document.querySelector('#leaf-submit');
const leafResult = document.querySelector('#leaf-result');
const imagePreview = document.querySelector('#image-preview');
const imagePreviewWrap = document.querySelector('#image-preview-wrap');
const leafPlaceholder = document.querySelector('#leaf-placeholder');
const uploadBox = document.querySelector('#upload-box');
let previewUrl = null;

function showMessage(element, text, isError = false) {
  element.hidden = false;
  element.className = `result-panel ${isError ? 'status-error' : 'status-info'}`;
  element.textContent = text;
}

function addText(parent, tag, text, className = '') {
  const child = document.createElement(tag);
  child.textContent = text;
  if (className) child.className = className;
  parent.append(child);
  return child;
}

async function loadCrops() {
  try {
    const response = await fetch('/api/crops');
    if (!response.ok) throw new Error('Crop model could not be loaded.');
    const data = await response.json();
    for (const crop of data.crops) {
      const option = document.createElement('option');
      option.value = crop;
      option.textContent = crop;
      targetCrop.append(option);
    }
  } catch (error) {
    targetCrop.disabled = true;
    targetCrop.options[0].textContent = 'Crop list unavailable';
  }
}
loadCrops();

cropForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  cropResult.hidden = false;
  cropResult.className = 'result-panel status-info';
  cropResult.textContent = 'Analysing the entered conditions…';
  const data = Object.fromEntries(new FormData(cropForm).entries());
  for (const field of ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']) data[field] = Number(data[field]);
  if (!data.target_crop) data.target_crop = null;
  try {
    const response = await fetch('/api/crop-predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Crop analysis is unavailable.');
    cropResult.replaceChildren(); cropResult.className = 'result-panel status-info';
    const head = document.createElement('div'); head.className = 'result-heading';
    const intro = document.createElement('div');
    addText(intro, 'h3', `Top model match: ${result.top_recommendation}`);
    addText(intro, 'p', 'Highest-scoring crop class for the supplied readings.');
    head.append(intro); addText(head, 'span', `${(result.recommendations[0].probability*100).toFixed(1)}%`, 'probability-chip'); cropResult.append(head);
    const list = document.createElement('div'); list.className = 'recommendation-list';
    for (const item of result.recommendations) {
      const row = document.createElement('div'); row.className = 'recommendation-row';
      addText(row, 'strong', item.crop);
      const bar = document.createElement('div'); bar.className = 'bar'; const fill = document.createElement('span'); fill.style.width = `${Math.max(2,item.probability*100)}%`; bar.append(fill); row.append(bar);
      addText(row, 'span', `${(item.probability*100).toFixed(1)}%`); list.append(row);
    }
    cropResult.append(list);
    if (result.selected_crop && result.selected_crop_probability !== null) addText(cropResult, 'p', `${result.selected_crop}: ${result.risk_band} (${(result.selected_crop_probability*100).toFixed(1)}% model probability).`, 'risk-chip');
    addText(cropResult, 'p', result.model_note, 'model-caveat');
  } catch (error) { showMessage(cropResult, error.message || 'Crop analysis failed. Please try again.', true); }
});

leafInput.addEventListener('change', () => {
  const file = leafInput.files?.[0];
  leafResult.hidden = true;
  leafButton.disabled = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  if (!file) { imagePreviewWrap.hidden = true; leafPlaceholder.hidden = false; return; }
  const allowed = ['image/jpeg','image/png'];
  if (!allowed.includes(file.type) || file.size > 8*1024*1024) {
    showMessage(leafResult, file.size > 8*1024*1024 ? 'The photo must be 8 MB or smaller.' : 'Choose a JPG or PNG image.', true);
    leafInput.value = ''; imagePreviewWrap.hidden = true; leafPlaceholder.hidden = false; return;
  }
  previewUrl = URL.createObjectURL(file); imagePreview.src = previewUrl;
  imagePreviewWrap.hidden = false; leafPlaceholder.hidden = true; leafButton.disabled = false;
});

leafButton.addEventListener('click', async () => {
  const file = leafInput.files?.[0]; if (!file) return;
  leafButton.disabled = true; leafButton.textContent = 'Checking photo…';
  showMessage(leafResult, 'Running the trained leaf classifier…');
  const body = new FormData(); body.append('file', file);
  try {
    const response = await fetch('/api/leaf-predict', {method:'POST',body});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Leaf classification is unavailable.');
    leafResult.replaceChildren(); leafResult.className = `result-panel ${result.low_confidence ? 'status-error' : 'status-info'}`;
    const head = document.createElement('div'); head.className = 'result-heading'; const intro = document.createElement('div');
    addText(intro, 'h3', result.low_confidence ? 'Low-confidence result — check with an expert' : result.prediction);
    addText(intro, 'p', result.low_confidence ? 'The model could not confidently identify a class.' : 'Top class predicted by the trained image model.');
    head.append(intro); addText(head, 'span', `${(result.confidence*100).toFixed(1)}%`, 'probability-chip'); leafResult.append(head);
    const list = document.createElement('div'); list.className = 'recommendation-list';
    for (const item of result.top_predictions) { const row=document.createElement('div'); row.className='recommendation-row'; addText(row,'strong',item.class_name); const bar=document.createElement('div'); bar.className='bar'; const fill=document.createElement('span'); fill.style.width=`${Math.max(2,item.probability*100)}%`; bar.append(fill); row.append(bar); addText(row,'span',`${(item.probability*100).toFixed(1)}%`); list.append(row); }
    leafResult.append(list); addText(leafResult,'p',result.model_note,'model-caveat');
  } catch (error) { showMessage(leafResult, error.message || 'Leaf classification failed. Please try again.', true); }
  finally { leafButton.disabled = false; leafButton.innerHTML = 'Classify leaf <span aria-hidden="true">↗</span>'; }
});

uploadBox.addEventListener('dragover', (event) => { event.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (event) => {
  event.preventDefault(); uploadBox.classList.remove('dragover');
  const file = event.dataTransfer.files?.[0]; if (!file) return;
  const transfer = new DataTransfer(); transfer.items.add(file); leafInput.files = transfer.files; leafInput.dispatchEvent(new Event('change'));
});
