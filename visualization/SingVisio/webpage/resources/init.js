/**
 * Copyright (c) 2023 Amphion.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// === init data ===
// load data from config.js
let config;

const loadConfig = (data) => {
    config = data;
    // ==== Init Display ====
    currentMode = Object.keys(config.pathData)[0];
    currentShowingPic = config.picTypes[3];
    initAlert();
    initSelect("mode_id", Object.keys(config.pathData), 0);
    refreshOptions();
    initSelect("pic_id", config.picTypes, 3, -1);
    drawStepMap(baseLink + "/data/mp_all/step_encoder_output.csv");
    updatePlayIcon(playMode)

    drawHistogram(config.histogramData.filter((e) => e.type === "high"), "#histogram", 128, "Metrics", "Score", "left");
    drawHistogram(config.histogramData.filter((e) => e.type === "low"), "#histogram2", 158, "", "Log score", "right");

    resetDisplay();

    defaultConfig();
}
const defaultConfig = () => {
    setTimeout(() => {
        selectStep(999);
        selectStep(100);
        selectStep(10);
    }, 1000)

    if (localStorage.getItem('GUIDED') === 'true') return;
    localStorage.setItem('GUIDED', 'true');

    const driver = window.driver.js.driver;

    const driverObj = driver({
        showProgress: true,
        steps: [
            { element: '#performance', popover: { title: 'Model Performance', description: 'Check the metrics of the model. Click rectangular to filter the best performance case.' } },
            { element: '#control_panel', popover: { title: 'Control Panel', description: 'Config mode and select data here.' } },
            { element: '#step_axis', popover: { title: 'Select Steps', description: 'Drag it to see the forming process of voice. Mel Spectrogram for selected steps displays on middle-top. Steps will be highlighted in left map correspondingly.' } },
            { element: '#step_preview', popover: { title: 'Preview Mel Spectrogram', description: 'Drap the "Step Axis" to select a step. Drag on graph to zoom in. Click right button to reset or click a step to pin for comparison.' } },
            { element: '#touch_map', popover: { title: 'Step Map', description: 'Click steps and add them to comparison area in the right.', }, },
            { element: '#display_10', popover: { title: 'Step Detail', description: 'You can choose up to 4 steps. Close it by click second right-top button.' } },
            { element: '#mel_10', popover: { title: 'Step\'s Mel Spectrogram', description: 'Drag to zoom in. Reset by click first right-top button.', }, }
        ]
    });

    driverObj.drive();
}
const initConfig = (path) => {
    fetch(path).then(response => response.json()).then(data => loadConfig(data)).catch((e) => {
        console.error(e);
        alert("Failed to load config file. Please check your network and try again.")
    });
}

// === shortcut ===

const $$ = (id) => document.getElementById(id)

// === init symbol ===
const circleD3 = d3.symbol();
const triangleD3 = d3.symbol().type(d3.symbolTriangle);

const refreshIcon = '<svg class="w-3.5 h-3.5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 18 20"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 1v5h-5M2 19v-5h5m10-4a8 8 0 0 1-14.947 3.97M1 10a8 8 0 0 1 14.947-3.97"/></svg>';
const closeIcon = '<svg class="w-3 h-3 m-[1px]" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6"/></svg>'

// === init global variable ===
let isAlertShown = false;
let targetSingers = [], sourceSongs, sourceSingers;
let displaySteps = [];
let currentMode, currentSong, currentTargetSinger, currentSinger;
let currentShowingPic = null;
let currentTextShow = false;
let enableReference = true;

let usedColorList = [];

let charts = [];
let darkMode = false;
let zoomStart, zoomEnd;

let playMode = (localStorage.getItem('AUTO_PLAY') ?? "true") === "true";

let currentCard
const $mel = $$("mel_card_container")
const $range = $$("range")

let dropdowns = []

let currentHistogram = [];

let highlightStep, resetStep, hoverStep;

let changeVideoTimer = [];

let compareNum = 0;

let lastDownload;
let hoveredStep = []
let lineChangeInterval;

let slow_mode_count = 0;

let jumpStep = 5;

