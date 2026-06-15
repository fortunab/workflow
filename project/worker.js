self.onmessage = async (event) => {
  const { prompt } = event.data;

  const sendTrace = (message) => self.postMessage({ type: 'trace', content: message });
  const sendChunk = (message) => self.postMessage({ type: 'chunk', content: message });

  sendTrace(`PROMPT RECEIVED: ${prompt}`);

  // Phase 1: Classifier
  sendTrace('MODELS[0]: Running Classifier (ResNet50 or Bionica)...');
  await delay(800);
  const clsResult = 'Positive for Polyp (Score: 0.992)';
  sendTrace(`RESULT: ${clsResult}`);

  // Phase 2: Object Detection
  sendTrace('MODELS[1]: Running Detector (YOLOv8)...');
  await delay(1000);
  const detResult = 'BBox Found: [43, 220, 110, 150]';
  sendTrace(`RESULT: ${detResult}`);

  // Phase 3: Segmentation
  sendTrace('MODELS[2]: Running Segmentor (SAM2 or SAM3)...');
  await delay(1200);
  const segResult = 'Mask Generated. Dice Score: 0.91. Area: 4.2% of frame.';
  sendTrace(`RESULT: ${segResult}`);

  // Phase 4: LLM Explainer mock
  sendTrace('MODELS[3]: Orchestrating results with LLM Explainer...');
  await delay(500);

  const mockAIReply = [
    'Final Report:',
    'The analysis confirms a polyp in the Kvasir-SEG frame.',
    'The detector localized the lesion with high confidence, and the segmentation mask indicates irregular borders with a Dice score of 0.91.',
    'Clinically, this appears consistent with a sessile polyp.',
    'Recommendation: optical biopsy and follow-up colonoscopy.'
  ].join(' ');

  const words = mockAIReply.split(' ');
  for (const word of words) {
    sendChunk(`${word} `);
    await delay(40);
  }

  self.postMessage({ type: 'done' });
};

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
