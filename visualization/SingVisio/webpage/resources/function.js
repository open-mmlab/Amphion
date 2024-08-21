/**
 * Copyright (c) 2023 Amphion.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2021, Observable Inc.
// Released under the ISC license.
// https://observablehq.com/@d3/color-legend
function Swatches(color, {
    columns = null,
    format,
    unknown: formatUnknown,
    swatchSize = 15,
    swatchWidth = swatchSize,
    swatchHeight = swatchSize,
    marginLeft = 0
} = {}) {
    const id = `-swatches-${Math.random().toString(16).slice(2)}`;
    const unknown = formatUnknown == null ? undefined : color.unknown();
    const unknowns = unknown == null || unknown === d3.scaleImplicit ? [] : [unknown];
    const domain = color.domain().concat(unknowns);
    if (format === undefined) format = x => x === unknown ? formatUnknown : x;

    function entity(character) {
        return `&#${character.charCodeAt(0).toString()};`;
    }

    if (columns !== null) return htl.html`<div style="display: flex; align-items: center; margin-left: ${+marginLeft}px; min-height: 33px; font: 10px sans-serif;">
  <style>

.${id}-item {
  break-inside: avoid;
  display: flex;
  align-items: center;
  padding-bottom: 1px;
}

.${id}-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: calc(100% - ${+swatchWidth}px - 0.5em);
}

.${id}-swatch {
  width: ${+swatchWidth}px;
  height: ${+swatchHeight}px;
  border-radius: ${+swatchHeight}px;
  margin: 0 0.5em 0 0;
}

  </style>
  <div style=${{ width: "100%", columns }}>${domain.map(value => {
        const label = `${format(value)}`;
        return htl.html`<div class=${id}-item>
      <div class=${id}-swatch style=${{ background: color(value) }}></div>
      <div class=${id}-label title=${label}>${label}</div>
    </div>`;
    })}
  </div>
</div>`;

    return htl.html`<div style="display: flex; align-items: center; min-height: 33px; margin-left: ${+marginLeft}px; font: 10px sans-serif;">
  <style>

.${id} {
  display: inline-flex;
  align-items: center;
  margin-right: 1em;
}

.${id}::before {
  content: "";
  width: ${+swatchWidth}px;
  height: ${+swatchHeight}px;
  margin-right: 0.5em;
  background: var(--color);
}

  </style>
  <div>${domain.map(value => htl.html`<span class="${id}" style="--color: ${color(value)}">${format(value)}</span>`)}</div>`;
}

function ramp(color, n = 256) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = n;
    const context = canvas.getContext("2d");
    for (let i = 0; i < n; ++i) {
        context.fillStyle = color(i / (n - 1));
        context.fillRect(0, n - i, 1, 1);
    }
    return canvas;
}

function legend({
    color,
    title,
    tickSize = 6,
    width = 36 + tickSize,
    height = 320,
    marginTop = 10,
    marginRight = 10 + tickSize,
    marginBottom = 20,
    marginLeft = 5,
    ticks = height / 64,
    tickFormat,
    tickValues
} = {}) {

    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("overflow", "visible")
        .style("display", "block");

    let tickAdjust = g => g.selectAll(".tick line").attr("x1", marginLeft - width + marginRight);
    let x;

    if (color.interpolator) {
        x = Object.assign(color.copy()
            .interpolator(d3.interpolateRound(height - marginBottom, marginTop)),
            { range() { return [height - marginBottom, marginTop]; } });

        svg.append("image")
            .attr("x", marginLeft)
            .attr("y", marginTop)
            .attr("width", width - marginLeft - marginRight)
            .attr("height", height - marginTop - marginBottom)
            .attr("preserveAspectRatio", "none")
            .attr("xlink:href", ramp(color.interpolator()).toDataURL());

        // scaleSequentialQuantile doesn’t implement ticks or tickFormat.
        if (!x.ticks) {
            if (tickValues === undefined) {
                const n = Math.round(ticks + 1);
                tickValues = d3.range(n).map(i => d3.quantile(color.domain(), i / (n - 1)));
            }
            if (typeof tickFormat !== "function") {
                tickFormat = d3.format(tickFormat === undefined ? ",f" : tickFormat);
            }
        }
    }

    svg.append("g")
        .attr("transform", `translate(${width - marginRight},0)`)
        .call(d3.axisRight(x)
            .ticks(ticks, typeof tickFormat === "string" ? tickFormat : undefined)
            .tickFormat(typeof tickFormat === "function" ? tickFormat : undefined)
            .tickSize(tickSize)
            .tickValues(tickValues))
        .call(tickAdjust)
        .call(g => g.select(".domain").remove())
        .call(g => g.append("text")
            .attr("x", marginLeft - width + marginRight)
            .attr("y", height - 2)
            .attr("fill", "currentColor")
            .attr("text-anchor", "start")
            // .attr("font-weight", "bold")
            .attr("class", "title")
            .text(title));

    return svg.node();
}


// === init info functions ===
const autoMapFunc = (id) => config.mapToName[id] ?? config.mapToSong[id] ?? id;
const mapToNameFunc = (id) => config.mapToName[id] ?? id;
const mapToSongFunc = (id) => config.mapToSong[id] ?? id;
const mapToSpaceFunc = (id) => config.mapToSpace[id] ?? id;
const isConfirmed = () => currentSong.length > 0 && currentSinger.length > 0 && currentTargetSinger.length > 0

const isSupportMultiMode = (id = 'all') => {
    const multiConfig = config.pathData[currentMode].multi;
    if (multiConfig === true) return true;
    if (multiConfig === false) return false;
    return multiConfig.includes(id);
}

const isMultiMode = () => currentSong.length >= 2 || currentSinger.length >= 2 || currentTargetSinger.length >= 2

const isSelectable = () => {
    return isMultiMode() || currentMode === "Metric Comparison"
}

const getSrcPerfix = (i = currentSinger[0], t = currentTargetSinger[0], s = currentSong[0]) => {
    let basePath = config.pathData[currentMode].data.find((d) => Object.keys(d.pathMap).includes(i)).basePath;
    return `${baseLink}/${basePath}/to_${t}/${i}_${s}_pred_autoshift`
}

const getCsvSrc = (v = 999, i = currentSinger[0], t = currentTargetSinger[0], s = currentSong[0]) => {
    return getSrcPerfix(i, t, s) + `_step_${v}.csv`
}

const getReferenceCsvSrc = (i = currentSinger[0], s = currentSong[0]) => `${baseLink}/data/rf_all/${i}_${s}.csv`

const getTargetReferenceCsvSrc = (i = currentSinger[0], s = currentSong[0]) => {
    const referenceMap = config.pathData[currentMode].referenceMap;
    let index = getSongs(currentSinger[currentSinger.length - 1]).indexOf(s);
    if (index > referenceMap[i].length) {
        index = referenceMap[i].length - 1;
    }
    const path = referenceMap[i][index];
    return `${baseLink}/data/rf_all/${path}.csv`
}

const getStepSrc = (s = currentSinger[0], t = currentTargetSinger[0], o = currentSong[0], p = currentShowingPic) => {
    return getSrcPerfix(s, t, o) + `_${p}_all_steps.csv`
}

const getMetricsSrc = (s = currentSinger[0], t = currentTargetSinger[0], o = currentSong[0]) => {
    return getSrcPerfix(s, t, o) + `_metrics.csv`
}

const getSongs = (singer) => {
    return config.pathData[currentMode].data.find((d) => Object.keys(d.pathMap).includes(singer))?.pathMap[singer].songs ?? [];
}

const findCorrespondingSong = (targetSinger, song = currentSong[0]) => {
    const singer = currentSinger[currentSinger.length - 1]; // always use the last one
    // if (singer === targetSinger) return song;Source Singer
    const index = getSongs(singer).indexOf(song); // find index
    const newSong = getSongs(targetSinger)[index];
    // console.log(`findCorrespondingSong: ${song} -> ${newSong}, index: ${index}, singer: ${singer}, targetSinger: ${targetSinger}, getSongs: ${getSongs(singer)}`)
    return newSong;
}

const getMultipleLable = () => {
    if (currentSinger.length > 1) {
        return ['sourcesinger', currentSinger]
    } else if (currentTargetSinger.length > 1) {
        return ['target', currentTargetSinger]
    } else if (currentSong.length > 1) {
        return ['song', currentSong]
    }
    return [null, null]
}

const autoLog = (value) => {
    // take log of value if it is too large
    if (value > 2) {
        return Math.log(value) / Math.log(4);
    }
    return value;
}

// === init UI bind functions ===
const bindVideo = (refId) => {
    let timer;
    const $video = $$(`video${refId}`)
    $video?.addEventListener('play', () => {
        clearInterval(timer)
        timer = setInterval(() => {
            const currentTime = $video.currentTime;
            charts.find((e) => e.id === refId)?.reset()
            charts.find((e) => e.id === refId)?.highlight(currentTime)
        }, 10)
    });
    $video?.addEventListener('pause', () => {
        clearInterval(timer)
        charts.find((e) => e.id === refId)?.reset()
    });
    $video?.addEventListener('ended', () => {
        clearInterval(timer)
        charts.find((e) => e.id === refId)?.reset()
    });
}

const bindIcon = (refId, svg, color) => {
    d3.select(`#icon${refId}`)
        .attr("width", 20)
        .attr("height", 20)
        .append("path")
        .attr("d", svg.size(120))
        .style("stroke", "white")
        .style("stroke-width", "2px")
        .attr("transform", `translate(10, 11)`)
        .style("fill", color)
}

const bindDiv = (refId, data = [], svgObject = null, color = "#000", close = () => { }, width = 345, height = 200) => {
    // trying to bind if exist: close, select, refresh, video, svg

    // bind close and select event
    $$(`close${refId}`)?.addEventListener('click', () => {
        close()
    })
    $$(`select${refId}`)?.addEventListener('change', (e) => {
        const checked = e.target.checked;
        // mark this chart as selected in charts
        charts.find((c) => c.id === refId).selected = checked;
        checkCompare()
    });
    // bind refresh event
    $$(`refresh${refId}`)?.addEventListener('click', () => {
        charts.forEach(c => c.sync && c.reset())
    })

    // when video is playing, highlight the chart
    bindVideo(refId)

    // draw svg using d3
    if (svgObject) bindIcon(refId, svgObject, color);

    // data struct:
    // [
    //     {col1: 0.1, col2: 0.2, ...(x axis)},
    //     {col1: 0.1, col2: 0.2, ...},
    //     ...(y axis),
    //     columns: ['col1', 'col2', ...]
    // ]

    if (data.length === 0) return;

    const arrayData = data.map(row => Object.values(row).map(d => +d));
    // draw mel spectrogram
    plotMelSpectrogram(arrayData, refId, color, close, width, height, true, hidePitch);

    charts.find((e) => e.id === refId)?.zoom(zoomStart, zoomEnd)
}

const getDiv = (refId, src, color, title, subtitle, svg = false, card = true) => {
    const div = document.createElement('div');
    const draggable = (isMultiMode() || userMode === "basic") ? '' : 'draggable="true"';
    div.innerHTML = (card ? `<div class="card min-w-[305px] p-2 w-full flex flex-col gap-1" id="display${refId}" ${draggable}>` : '<div>') +
        `<div class="flex items-center">` +
        `<input id="select${refId}" type="checkbox" value="" class="checkbox mr-1">` +
        (svg ? `<svg id="icon${refId}" class="shrink-0 h-[20px] w-[20px]"></svg>` : '') +
        `<div class="flex flex-col ml-1 mr-1">` +
        `<h5 class="text-base font-bold tracking-tight mb-0 text-[${color}] line-clamp-1" id="title${refId}">${title}</h5>` +
        `<h5 class="text-xs tracking-tight mb-0 text-[${color}] line-clamp-1">${subtitle}</h5>` +
        `</div>` +
        (card ? `<div class="flex flex-row ml-auto">` +
            `<a class="btn-sec h-9 w-9 p-2.5 mb-0" id="refresh${refId}">${refreshIcon}</a>` +
            `<a class="btn-sec h-9 w-9 p-2.5 mb-0" id="close${refId}">${closeIcon}</a>` +
            `</div>` : '') +
        `</div>` +
        `<div class="mx-auto min-h-[200px]" id="mel${refId}">${loadingDiv}</div>` +
        `<audio class="w-full" id="video${refId}" controls src="${src}"></audio>` +
        `</div>`;
    return div.firstChild;
}

const cachedDifference = {}


const getColor = (color) => {
    if (color > 1) {
        color = 1
    }
    if (color < 0) {
        color = 0
    }
    return d3.interpolateBlues(color)
}

const calculateStepDifference = (step1, step2, td) => {
    // return a number to represent the difference between two steps
    // the larger the number, the more different between two steps
    const csv1 = getCsvSrc(step1);
    const csv2 = getCsvSrc(step2);
    const key = `${currentSinger[0]}-${currentTargetSinger[0]}-${currentSong[0]}`
    let diff = 0;
    if (cachedDifference[`${key}-${step1}-${step2}`] || cachedDifference[`${key}-${step2}-${step1}`]) {
        diff = cachedDifference[`${key}-${step1}-${step2}`] ?? cachedDifference[`${key}-${step2}-${step1}`];
        td.style.backgroundColor = getColor(diff);
        return;
    }
    d3.csv(csv1, (data1) => {
        const arrayData1 = data1.map(row => Object.values(row).map(d => +d));
        d3.csv(csv2, (data2) => {
            const arrayData2 = data2.map(row => Object.values(row).map(d => +d));
            // console.log(arrayData1, arrayData2)
            arrayData1.forEach((d1, i) => {
                const d2 = arrayData2[i];
                d1.forEach((v1, j) => {
                    const v2 = d2[j];
                    diff += (v1 - v2) ** 2;
                })
            })
            const normalizedDiff = Math.sqrt(diff) / 6000;
            cachedDifference[`${key}-${step1}-${step2}`] = normalizedDiff;
            td.style.backgroundColor = getColor(normalizedDiff);
        })
    })
    // console.log(`step difference between ${step1} and ${step2} is ${diff}`)
    // return diff;
}

let showTooltipFn = null
const showTooltip = (content) => {
    // show a tool tip around the mouse
    const tooltip = $$("tooltip");
    tooltip.innerHTML = content;
    tooltip.classList.remove("invisible");
    showTooltipFn = (e) => {
        tooltip.style.left = `${e.pageX + 10}px`;
        tooltip.style.top = `${e.pageY + 10}px`;
    }
    document.addEventListener("mousemove", showTooltipFn)
}
const hideTooltip = () => {
    $$("tooltip").classList.add("invisible");
    document.removeEventListener("mousemove", showTooltipFn);
}

const getGridDiv = () => {
    // a table with all selected steps lined up in cols and rows with step number
    const table = document.createElement('table');
    table.classList.add('table-auto', 'h-fit', 'w-fit', 'border-collapse', 'border', 'border-gray-200', 'dark:border-gray-700');
    for (let i = -1; i < gridSelected.length; i++) {
        const tr = document.createElement('tr');
        tr.classList.add('border', 'border-gray-200', 'dark:border-gray-700');
        for (let j = -1; j < gridSelected.length; j++) {
            const td = document.createElement('td');
            td.classList.add('border', 'border-gray-200', 'dark:border-gray-700');

            if (i === -1 || j === -1) {
                td.style.width = '35px';
                td.style.minWidth = '35px';
                td.style.height = '35px';
                td.style.textAlign = 'center';
                // table head
                if (i === -1 && j === -1) {
                    tr.appendChild(td);
                    continue;
                }
                if (i === -1) {
                    td.innerText = `${gridSelected[j]}`;
                    tr.appendChild(td);
                    continue;
                }
                if (j === -1) {
                    td.innerText = `${gridSelected[i]}`;
                    tr.appendChild(td);
                    continue;
                }
                continue;
            }

            const step = gridSelected[i];
            const step2 = gridSelected[j];

            if (step === step2) {
                // td.innerText = `N/A`;
                td.style.backgroundColor = getColor(0);
                tr.appendChild(td);
                continue;
            }
            calculateStepDifference(step, step2, td);
            td.style.cursor = 'pointer';

            td.addEventListener('mouseover', () => {
                showTooltip(`Step ${step} vs Step ${step2}`)
            })
            td.addEventListener('mouseout', () => {
                hideTooltip()
            })

            td.addEventListener('click', () => {
                if (downloadingLock.length !== 0) {
                    alert('Operating too fast, please wait for the previous operation to finish') // fix removing bug
                    return
                }
                gridComparison.forEach((step) => {
                    charts.filter(c => c.id.endsWith(step))?.forEach(c => c.close(true)); // close all previous comparison display
                })

                // gridComparison = [`${step}`, `${step2}`];
                gridComparison.push(`${step}`);
                gridComparison.push(`${step2}`);
                selectStep(step);
                selectStep(step2);
            })

            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
    // get outter div width and height, then scale the table
    if (gridSelected.length >= 6) {
        const tableHeight = 35 * (gridSelected.length + 1) + 1;
        const divHeight = 250;
        const factor = divHeight / tableHeight;
        console.log(factor)
        table.style.transform = `scale(${factor})`;
    } else {
        table.style.transform = `scale(1)`;
    }
    return table;
}

const loadGrid = () => {
    // generate two divs for comparison display: display${refId}, display${refId}. Undraggable, selectable, unclosable
    selectStep(-1)
    // gridSelected = [0, 10, 50, 100, 200, 999]; // default
    // gridSelected = Array.from({ length: 41 }, (v, i) => i * 5); // 0 - 200
    // gridSelected = Array.from({ length: 21 }, (v, i) => i * 10); // 0 - 200
    // gridSelected = Array.from({ length: 21 }, (v, i) => i * 10 + 200); // 200 - 400
    // gridSelected = Array.from({ length: 21 }, (v, i) => (i * 30 + 400 === 1000) ? 999 : i * 30 + 400); // 400 - 999
    gridSelected = [50, 250, 650, 850, 950]; // mid point + 950
    // gridSelected = [0, 100, 200, 300, 600, 700, 800, 900]; // start and end point
    // gridSelected = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 999]; // exponential
    // gridSelected = Array.from({ length: 10 }, (v, i) => Math.pow(2, i));
    // gridSelected = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987] // Fibonacci
    gridComparison = ["ph1", "ph2"];
    if (gridComparison.length > 2) {
        alert('System error: Only 2 steps can be compared in grid mode')
        return
    }
    gridComparison.forEach((step, i) => {
        selectStep(step)
    })

    // generate a div to save grid: grid_select
    // in the div, user can see a grid table with all selected steps lined up in cols and rows with step number
    // in each cell, user can see a color representing the difference between two steps (cols and rows steps)
    // user can click on the cell to select the step for comparison, which will be highlighted in the left map, and in the previous comparison display divs
    $$("grid_table").appendChild(getGridDiv())
    $$("grid_table_legend").appendChild(legend({
        color: d3.scaleSequential()
            .domain([0, 100])
            .interpolator(d3.interpolateBlues),
        width: 30,
        tickSize: 2,
        height: 250,
        title: "Difference",
        tickFormat: (d) => `${d}%`
    }))
}


const addComparison = (step) => {
    // add the step into the gridSelected then update the grid table with new added information
    // then rerender the grid table
    let $table = $$("grid_table").querySelector("table");
    step = +step;

    if (gridSelected.includes(step)) {
        // remove the step
        gridSelected = gridSelected.filter((s) => s !== step);
    } else {
        gridSelected.push(step);
        gridSelected = gridSelected.sort((a, b) => a - b);
    }

    const table = getGridDiv();
    $table.replaceWith(table);
    $table = table;
}


// === init UI component functions ===
const initInterface = () => {
    availableMode = Object.keys(config.pathData).filter((k) => config.pathData[k].users.includes(userMode));
    initSelect("mode_id", availableMode, 0);

    refreshOptions();

    if (userMode === "basic") {
        resetDisplay(true, false, true); // clear the pin area
        // loadGrid(); // load the grid component in the pin area
        // $$("components").classList.add("hidden");
    }
    if (userMode === "advanced") {
        resetDisplay(false, false, true);
        // $$("components").classList.remove("hidden");
        selectStep(999);
        selectStep(100);
        selectStep(10);
    }
    $$("mode_change").textContent = userMode === "basic" ? "Switch to Advanced" : "Switch to Basic";
    // set autoplay
    playMode = (localStorage.getItem('AUTO_PLAY') ?? "true") === "true";
    updatePlayIcon(playMode)
}
const initAlert = () => {
    const show = () => {
        isAlertShown = true;
        $$("alert").classList.remove("hidden");
        $$("alert").classList.add("flex");
    }
    const hide = () => {
        isAlertShown = false;
        $$("alert").classList.add("hidden");
        $$("alert").classList.remove("flex");
    }
    window.alert = (title, msg = title) => {
        $$("alert_title").innerText = title === msg ? "Alert" : title;
        $$("alert_text").innerHTML = msg.replaceAll("\n", "<br />");
        show();
        $$("finish_alert").focus();
    }

    $$("close_alert").addEventListener("click", () => {
        hide()
    })
    $$("finish_alert").addEventListener("click", () => {
        hide()
    })

    document.addEventListener("keydown", (e) => {
        if (isAlertShown && e.key === "Escape") {
            hide()
        }
    })
}
const initOptions = (id, options) => {
    if (id === "sourcesinger_id") {
        currentSinger = [options[0]]
    } else if (id === "song_id") {
        currentSong = [options[0]]
    } else if (id === "target_id") {
        currentTargetSinger = [options[0]]
    }

    const $dropdown = $$(`${id}_dropdown`)
    if (!dropdowns.includes(id)) {
        dropdowns.push(id)

        $$(id).addEventListener("click", (e) => {
            if ($dropdown.classList.contains("hidden")) {
                dropdowns.forEach((dd) => {
                    const d = $$(`${dd}_dropdown`)
                    if (!d.classList.contains("hidden")) {
                        d.classList.add("hidden")
                    }
                })
                $dropdown.classList.remove("hidden")
            } else {
                $dropdown.classList.add("hidden")
            }
            e.stopPropagation();
        });

        // Event listener for the blur event on the document
        document.addEventListener("click", (event) => {
            // Check if the clicked element is outside the dropdown
            if (!$dropdown.contains(event.target)) {
                if (!$dropdown.classList.contains("hidden")) {
                    $dropdown.classList.add("hidden")
                }
            }
        });

    }

    const $ul = $dropdown.querySelector("ul");

    $ul.innerHTML = '';
    options.forEach((o, i) => {
        const $li = document.createElement("li");
        $li.classList.add("px-4", "flex", "items-center", "hover:bg-gray-100", "cursor-pointer", "dark:hover:bg-gray-600", "dark:hover:text-white");
        const $input = document.createElement("input");
        $input.id = `${id}_${o}`;
        $input.checked = i === 0
        $input.type = "checkbox";
        $input.value = o;
        $input.classList.add("checkbox");
        const $label = document.createElement("label");
        $label.htmlFor = `${id}_${o}`;
        $label.classList.add("w-full", "py-2", "ml-2", "text-sm", "text-gray-900", "dark:text-gray-300");
        $label.innerText = autoMapFunc(o);
        $li.appendChild($input);
        $li.appendChild($label);
        $input.addEventListener("change", (e) => {
            // update current after click
            if (e.target.checked) {
                // enable
                if (id === "sourcesinger_id") {
                    if ((!isSupportMultiMode(id) || isMultiMode()) && currentSinger.length > 0) {
                        // already multi, cancel previous one
                        $$(`${id}_${currentSinger.shift()}`).checked = false;
                    }

                    if (!refreshOptions(false, o, currentSinger.length > 0 ? currentSinger[0] : null)) {
                        // no avaliable target singer
                        alert('no avaliable target singer')
                        return;
                    }

                    currentTargetSinger = [targetSingers[0]];
                    currentSinger.push(o);
                } else if (id === "song_id") {
                    if ((!isSupportMultiMode(id) || isMultiMode()) && currentSong.length > 0) {
                        $$(`${id}_${currentSong.shift()}`).checked = false;
                    }
                    currentSong.push(o);
                } else if (id === "target_id") {
                    if ((!isSupportMultiMode(id) || isMultiMode()) && currentTargetSinger.length > 0) {
                        $$(`${id}_${currentTargetSinger.shift()}`).checked = false;
                    }
                    currentTargetSinger.push(o);
                }
            } else if (id === "sourcesinger_id") {
                currentSinger = currentSinger.filter((s) => s !== o);
                if (currentSinger.length > 0) {
                    refreshOptions(false, currentSinger[0])
                }
            } else if (id === "song_id") {
                currentSong = currentSong.filter((s) => s !== o);
            } else if (id === "target_id") {
                currentTargetSinger = currentTargetSinger.filter((s) => s !== o);
            }
            resetDisplay();
        })
        $ul.appendChild($li);
    })
}

const initSelect = (id, content, defaultIndex, limit = -1) => {
    const dropdown = $$(id);
    dropdown.innerHTML = '';
    for (let i = 0; i < content.length; i++) {
        if (limit > 0 && i >= limit) break;
        const option = document.createElement("option");
        option.value = content[i];
        option.textContent = mapToSpaceFunc(content[i]);
        if (i === defaultIndex) option.selected = 1;
        dropdown.appendChild(option);
    }
}

const updateSelect = (id, content, key) => {
    const dropdown = $$(id);
    dropdown.selectedIndex = content.indexOf(key);
}

const updateOptions = (id, options) => {
    const $dropdown = $$(`${id}_dropdown`)
    const $ul = $dropdown.querySelector("ul");
    const $inputs = $ul.querySelectorAll("input");
    $inputs.forEach((i) => {
        if (options.includes(i.value)) {
            i.checked = true;
        } else {
            i.checked = false;
        }
    })
    const $button = $$(`${id}_text`)
    options = options.map((o) => autoMapFunc(o))
    if (options.length === 0) {
        $button.innerHTML = `<span class="text-red-500">Please select</span>`;
    } else if (options.length === 1) {
        $button.innerHTML = `${options[0]}`;
    } else {
        $button.innerHTML = `${options[0]} + ${options[1]}`;
    }
}

const refreshOptions = (reset = true, selectedSourceSinger, selectedSourceSinger2 = null) => {
    if (reset) {
        sourceSingers = config.pathData[currentMode].data.map((d) => Object.keys(d.pathMap).flat()).flat();
        initOptions("sourcesinger_id", sourceSingers);
        selectedSourceSinger = selectedSourceSinger ?? sourceSingers[0];
    }

    sourceSongs = config.pathData[currentMode].data.map((d) => d.pathMap[selectedSourceSinger]?.songs).flat().filter((s) => s !== undefined);
    initOptions("song_id", sourceSongs); // update avaliable source songs


    const avaliableTargetSingers = config.pathData[currentMode].data.map((d) => d.pathMap[selectedSourceSinger]?.targets).flat().filter((s) => s !== undefined);
    // take intersection
    console.log(selectedSourceSinger, selectedSourceSinger2)
    if (selectedSourceSinger2) {
        const avaliableTargetSingers2 = config.pathData[currentMode].data.map((d) => d.pathMap[selectedSourceSinger2]?.targets).flat().filter((s) => s !== undefined);
        targetSingers = avaliableTargetSingers2.filter((value) => avaliableTargetSingers.includes(value));
    } else {
        targetSingers = avaliableTargetSingers
    }

    if (targetSingers === undefined || targetSingers.length === 0) {
        return false
    }
    initOptions("target_id", targetSingers); // update avaliable target singers
    return true
}

const lockOptions = () => {
    ["sourcesinger_id", "song_id", "target_id"].forEach((dd) => {
        const d = $$(dd)
        d.disabled = true;
    })
}

const unlockOptions = () => {
    ["sourcesinger_id", "song_id", "target_id"].forEach((dd) => {
        const d = $$(dd)
        d.disabled = false;
    })
}


const updatePlayIcon = (playMode) => {
    const $icon_play = $$("icon_play")
    const $icon_stop = $$("icon_stop")

    if (playMode) {
        $icon_play.style = "display: none";
        $icon_stop.style = "display: block";
    } else {
        $icon_play.style = "display: block";
        $icon_stop.style = "display: none";
    }
}

const drawHistogram = (data, id, width = 200, xlabel = "Metrics", ylable = "Performance", yside = "left") => {
    // shrink data to 0-1 by Z-score Normalization
    data = data.map((d, i) => {
        return {
            name: d.name,
            type: d.type,
            value: d.value,
            rawValue: d.value
        }
    })

    // Declare the chart dimensions and margins.
    // const width = 200;
    const height = isMultiMode() ? (currentShowingPic == "encoded_step" ? 180 : 136) : 160;
    const rectWidth = 15;
    const marginTop = 20;
    const marginRight = yside === "left" ? 30 : 60;
    const marginBottom = 15;
    const marginLeft = yside === "left" ? 40 : 10;

    // Declare the x (horizontal position) scale.
    const x = d3.scalePoint()
        .domain(data.map((d) => d.name))
        .range([width - marginRight, marginLeft + rectWidth])


    // Declare the y (vertical position) scale.
    let y
    if (yside === "left") {
        y = d3.scaleLinear()
            .domain([0, 1])
            .range([height - marginBottom, marginTop]);
    } else {
        // log
        y = d3.scaleLog()
            .domain([1, 100])
            .range([height - marginBottom, marginTop]);
    }

    // Create the SVG container.
    d3.select(id).html("");
    const svg = d3.select(id)
        .attr("style", `color: ${darkMode ? 'white' : 'black'};`)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", `max-width: 100%; height: auto;`);

    // Add a rect for each bin.
    svg.append("g")
        .selectAll()
        .data(data)
        .enter()
        .append("rect")
        .attr("fill", (_, i) => metricsColors[yside === "left" ? 1 - i : 4 - i])
        .attr("id", (d) => `rect_${d.name}`)
        .attr("x", (d) => x(d.name) - rectWidth / 2)
        .attr("width", (d) => rectWidth)
        .attr("y", (d) => y(d.value))
        .attr("height", (d) => height - marginBottom - y(d.value))
        .on("mouseover", (e) => {
            if (currentHistogram.includes(e)) return;
            d3.select(`#rect_${e.name}`).attr("stroke", "gray").attr("stroke-width", 2)
        })
        .on("mouseout", (e) => {
            if (currentHistogram.includes(e)) return;
            d3.select(`#rect_${e.name}`).attr("stroke", "none")
        })
        .on("click", (e) => {
            if (currentHistogram.includes(e)) {
                currentHistogram = currentHistogram.filter((s) => s !== e);
                d3.select(`#rect_${e.name}`).attr("stroke", "none")
            } else {
                const previous_e = currentHistogram.shift();
                if (previous_e) {
                    d3.select(`#rect_${previous_e.name}`).attr("stroke", "none")
                }
                currentHistogram.push(e);
                d3.select(`#rect_${e.name}`).attr("stroke", "currentColor").attr("stroke-width", 2)
            }
            if (currentHistogram.length > 0) {
                showBestCase();
            } else {
                unlockOptions();
            }
        })

    // Add a label for each bin.
    svg.append("g")
        .attr("fill", "currentColor")
        .attr("text-anchor", "middle")
        .attr("font-family", "sans-serif")
        .attr("font-size", 10)
        .selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .attr("id", (d) => `label_${d.name}`)
        .attr("x", (d) => x(d.name))
        .attr("y", (d) => y(d.value) - 4)
        .text((d) => d.rawValue.toFixed(2));

    // Add the x-axis and label.
    svg.append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(d3.axisBottom(x).tickSizeInner(1).tickSizeOuter(0))
        .call((g) =>
            g.select(".domain").attr("stroke", "currentColor")
                .attr("d", yside === "left" ? "M40.5,0V0.5H115.5V0" : "M10.5,0V0.5H115.5V0")
        )

    // Add the y-axis and label, and remove the domain line.
    if (yside === "left") {
        svg.append("g")
            .attr("transform", `translate(${marginLeft},0)`)
            .call(d3.axisLeft(y).ticks(height / 40))
            .call((g) => g.select(".domain").attr("stroke", "currentColor"))
            .call((g) => g.append("text")
                .attr("x", -height / 2 + ylable.length * 2)
                .attr("y", -marginLeft + 10)
                .attr("font-size", 12)
                .attr("fill", "currentColor")
                .attr("transform", "rotate(-90)")
                .text(ylable));
    } else {
        svg.append("g")
            .attr("transform", `translate(${width - marginRight + 15},0)`)
            .call(d3.axisRight(y).ticks(2).tickFormat(d3.format(".0f")))
            .call((g) => g.select(".domain").attr("stroke", "currentColor"))
            .call((g) => g.append("text")
                .attr("x", height / 2 - ylable.length * 2)
                .attr("y", -30)
                .attr("font-size", 12)
                .attr("fill", "currentColor")
                .attr("transform", "rotate(90)")
                .text(ylable));
    }

    svg.selectAll("text").attr("fill", "currentColor")
    svg.selectAll("line").attr("stroke", "currentColor")

    // Return the SVG element.
    return svg.node();
}

const createColorLegend = (svg) => {
    const width = +svg.attr('width'); // Get width from SVG attribute
    const height = +svg.attr('height'); // Fixed height for the legend

    svg.attr('height', height); // Set the height of the SVG

    // Set up the color scale using the Red-Yellow-Blue interpolator
    const colorScale = d3.scaleSequential()
        .domain([999, 0])
        .interpolator(d3.interpolateRdBu);

    // Gradient definition
    const defs = svg.append('defs');
    const linearGradient = defs.append('linearGradient')
        .attr('id', 'linear-gradient');

    // Define the gradient stops
    d3.range(0, 1.01, 0.01).forEach(t => {
        linearGradient.append('stop')
            .attr('offset', `${t * 100}%`)
            .attr('stop-color', colorScale(t * 1000));
    });

    // Draw the color rectangle
    svg.append('rect')
        .attr('width', width - 20)
        .attr('height', 8)
        .attr('x', 10)
        .attr('y', 2)
        .style('fill', 'url(#linear-gradient)');

    // Add an axis to the legend
    const xScale = d3.scaleLinear()
        .domain([999, 0])
        .range([10, width - 11]);

    // Custom ticks
    const ticks = [0, 250, 500, 750, 999];
    const axis = d3.axisBottom(xScale)
        .tickValues(ticks)
        .tickFormat(d3.format(".0f"))
        .tickSize(10)
        .tickPadding(1);

    svg.append('g')
        .attr('transform', `translate(0, 2)`)
        .call(axis)
        .call(g => g.select('.domain').remove())
        .selectAll('text')
        .attr('fill', darkMode ? "white" : "black")
        .attr('font-size', 8)
}


const drawContour = (data, SVG, width, height, id_i, x, y) => {
    is_large = id_i === "full";
    is_filtered = data.length < 1000;

    const contours = d3.contourDensity()
        .x(function (d) { return x(d.heng); })
        .y(function (d) { return y(d.shu); })
        .size([width, height])
        .cellSize(is_large ? 2 : 2)
        .bandwidth(is_large ? 5 : 3) // change bandwidth to adjust the contour density
        (data);


    const countourIndex = 1;

    const contour = SVG.append('g').attr("class", "zoom_g")

    contour.selectAll("path")
        .data([contours[countourIndex]])
        .enter().append("path")
        .attr("d", d3.geoPath())
        .attr("stroke", darkMode ? "white" : "black")
        .attr("fill", "none")

    // make a mask
    const mask = SVG.append('g').attr("class", "zoom_g")
        .append("mask")
        .attr("id", `contour_mask${id_i}`)

    mask.append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "black")

    mask.append("path")
        .attr("d", d3.geoPath()(contours[countourIndex]))
        .attr("fill", "white")

    // fill background color
    const backgroundScatters = SVG.append('g')
        .attr("class", "zoom_g")
        .attr("mask", `url(#contour_mask${id_i})`)

    backgroundScatters
        .selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", (d) => x(d.heng))
        .attr("cy", (d) => y(d.shu))
        .attr("r", is_large ? 12 : 6)
        .style("opacity", is_filtered ? 1 : 0.6) // opacity for countour background
        .style("fill", (d) => {
            return d3.interpolateRdBu(d.index / 1000);
        });
}

const drawScatter = (scatter_g, data, d, id_i, fill_color, x, y) => {
    is_large = id_i.includes("full");

    scatter_g.selectAll("circle")
        .data(data).enter()
        .append("path")
        .attr("d", d)
        .attr("id", (d) => `point${id_i}_${d.index}`)
        .attr("transform", (d) => `translate(${x(d.heng)}, ${y(d.shu)})`)
        .style("fill", fill_color)
        .attr("stroke", "white")
        .style("opacity", stepOpacity)
        .style("stroke-width", `${is_large ? stepStrokeWidth : stepStrokeWidth * 0.5}px`)
        .on("click", (d) => {
            const step = Number(d.index).toFixed(0)
            selectStep(step);
        })
        .on("mouseover", (d) => {
            const step = Number(d.index).toFixed(0)
            $range.value = 999 - step;
            lineChange(false);
        })
        .on("mouseout", (d) => {
            const step = Number(d.index).toFixed(0)
            if (hoveredStep.filter(s => s === step).length > 0) {
                hoveredStep = hoveredStep.filter(s => s !== step)
                resetStep(step)
            }
            $$("current_step_display_number").innerHTML = "";
        });
}

const drawStepMap = (csvPath, csvPath2 = '') => {
    // if csvPath2 is not null: compare mode, display two data in same figure
    // console.log(csvPath, csvPath2)
    const width = 280;
    const height = ((isMultiMode() && currentShowingPic != "encoded_step") ? 240 : 316);

    SVG = d3.select("#dataviz_axisZoom")
        .append("svg")
        .attr("width", width)
        .attr("height", height)

    // init a unzoomed SVG to display the legend
    const legendSVG = d3.select("#dataviz_axisZoom")
        .append("svg")
        .attr("id", "color_legend")
        .attr("width", 150)
        .attr("height", 25)
        .style("position", "absolute")
        .style("bottom", "0")
        .style("right", "0")

    createColorLegend(legendSVG)


    let data1, data2

    const downloadingData = () => {
        d3.csv(csvPath, (data) => {
            data1 = data;
            startDrawing()
        });
        if (csvPath2 !== "") d3.csv(csvPath2, (data) => {
            data2 = data;
            startDrawing()
        });
    }


    if (samplingSteps) {
        path = csvPath.split("/").pop()
        fetch(baseLink + `/process_map?input_path=${encodeURI(csvPath.replace(".csv", ".npy"))}&num_steps=${samplingNum}`)
            .then(response => response.json())
            .then(json => {
                console.log(json)
                sampledSteps = json.selected_steps
                downloadingData()
            })
            .catch((error) => {
                console.error('Error:', error);
                alert('Error: ' + error + ' Please try to reload the page')
            });
    } else {
        downloadingData()
    }

    const startDrawing = () => {
        if (!csvPath && !data1) {
            // lack of data, waiting all data loaded
            return;
        }
        if (csvPath2 !== "" && !data2) {
            // lack of data, waiting all data loaded
            return;
        }
        // console.log(data1, data2)
        const x = d3.scaleLinear()
            .domain([0, 1])
            .range([10, width - 10]);

        const y = d3.scaleLinear()
            .domain([0, 1])
            .range(isMultiMode() ? [height - 10, 10] : [height - 25, 10]);

        const zoom = d3.zoom()
            .scaleExtent([0.5, 25])
            .extent([[0, 0], [width, height]])
            .wheelDelta((e) => -event.deltaY * (event.deltaMode === 1 ? 0.05 : event.deltaMode ? 1 : 0.002) * (event.ctrlKey ? 10 : 1))
            .on("zoom", () => {
                SVG.selectAll(".zoom_g").attr("transform", d3.event.transform);
            });

        SVG
            .append("rect")
            .attr("width", width)
            .attr("height", height)
            .style("fill", "none")
            .style("pointer-events", "all");

        SVG.call(zoom);

        // draw line
        // SVG.append('g')
        //     .selectAll(".line")
        //     .data(data1)
        //     .enter()
        //     .append("line")
        //     .attr("x1", (d, i) => x(data1[i - 1]?.heng ?? d.heng))
        //     .attr("y1", (d, i) => y(data1[i - 1]?.shu ?? d.shu))
        //     .attr("x2", (d, i) => x(d.heng))
        //     .attr("y2", (d, i) => y(d.shu))
        //     .attr("stroke", "black")
        //     .attr("stroke-width", 1)
        //     .attr("opacity", 0.5)



        // here to filter data
        if (samplingSteps) {
            data1 = data1.filter((d) => sampledSteps.includes(d.index))
            if (data2) {
                data2 = data2.filter((d) => sampledSteps.includes(d.index))
            }
        }

        if (data1 && !data2) drawContour(data1, SVG, width, height, "full", x, y)


        if (data1) {
            const scatter = SVG.append('g').attr("class", "zoom_g")
            drawScatter(scatter, data1, circleD3.size(64), "full", "#61a3a9", x, y)
        }
        if (data2) {
            const scatter2 = SVG.append('g').attr("class", "zoom_g")
            drawScatter(scatter2, data2, triangleD3.size(64), "full2", "#a961a3", x, y)
        }
        if (data1 && data2) {
            // create two small SVG
            const width2 = width / 2;
            const height2 = height / 2;
            const SVG1 = d3.select("#dataviz_axisZoom").append("svg")
                .attr("width", width2)
                .attr("height", height2)

            const SVG2 = d3.select("#dataviz_axisZoom").append("svg")
                .attr("width", width2)
                .attr("height", height2)

            const x2 = d3.scaleLinear().domain([-0.1, 1.1]).range([0, width2]);
            const y2 = d3.scaleLinear().domain([-0.1, 1.1]).range([height2, 0]);

            drawContour(data1, SVG1, width2, height2, "1", x2, y2)
            drawContour(data2, SVG2, width2, height2, "2", x2, y2)

            const scatter1 = SVG1.append('g').attr("class", "zoom_g")
            drawScatter(scatter1, data1, circleD3.size(24), "", "#61a3a9", x2, y2)

            const scatter2 = SVG2.append('g').attr("class", "zoom_g")
            drawScatter(scatter2, data2, triangleD3.size(24), "2", "#a961a3", x2, y2)

            // update the position and size of color legend
            d3.select("#color_legend")
                .style("right", "68px")
                .style("bottom", "-8px")
                .style("transform", "scale(0.7)")
        } else {
            d3.select("#color_legend")
                .style("right", "0")
                .style("bottom", "-2px")
                .style("transform", "scale(1)")
        }

        hoverStep = (step) => {
            step = `${step}`
            if (displaySteps.includes(step)) {
                d3.select(`#pointfull_${step}`).raise();
                d3.select(`#pointfull2_${step}`).raise();
                d3.select(`#point_${step}`).raise();
                d3.select(`#point2_${step}`).raise();
                return;
            }
            const color = (isMultiMode()) ? "#FFA500" : "#ff00ed"
            const color2 = (isMultiMode()) ? "#1C64F2" : "#ff00ed"
            d3.select(`#pointfull_${step}`)
                .style("fill", color)
                .style("stroke", "white")
                .style("stroke-width", `${stepStrokeWidth * 2}px`)
                .attr("d", circleD3.size(192))
                .style("cursor", "pointer")
                .style("opacity", 1)
                .raise(); // move to front
            if (data2) d3.select(`#pointfull2_${step}`)
                .style("fill", color2)
                .style("stroke", "white")
                .style("stroke-width", `${stepStrokeWidth * 2}px`)
                .attr("d", triangleD3.size(192))
                .style("opacity", 1)
                .style("cursor", "pointer")
                .raise(); // move to front
            if (data1 && data2) {
                d3.select(`#point_${step}`)
                    .style("fill", color)
                    .style("stroke", "white")
                    .style("stroke-width", `${stepStrokeWidth}px`)
                    .attr("d", circleD3.size(64))
                    .style("cursor", "pointer")
                    .style("opacity", 1)
                    .raise(); // move to front
                d3.select(`#point2_${step}`)
                    .style("fill", color2)
                    .style("stroke", "white")
                    .style("stroke-width", `${stepStrokeWidth}px`)
                    .attr("d", triangleD3.size(64))
                    .style("cursor", "pointer")
                    .style("opacity", 1)
                    .raise(); // move to front
            }
            $$("current_step_display_number").innerText = step;
        }

        highlightStep = (step, color = "#000", color2 = color) => {
            d3.select(`#pointfull_${step}`)
                .style("fill", color)
                .attr("d", circleD3.size(192))
                .style("stroke", "white")
                .style("stroke-width", `${stepStrokeWidth * 2}px`)
                .style("cursor", "pointer")
                .style("opacity", 1)
                .raise(); // move to front
            if (data2) d3.select(`#pointfull2_${step}`)
                .style("fill", color2)
                .attr("d", triangleD3.size(192))
                .style("stroke", "white")
                .style("stroke-width", `${stepStrokeWidth * 2}px`)
                .style("cursor", "pointer")
                .style("opacity", 1)
                .raise(); // move to front
            if (data1 && data2) {
                d3.select(`#point_${step}`)
                    .style("fill", color)
                    .attr("d", circleD3.size(24))
                    .style("stroke", "white")
                    .style("stroke-width", `${stepStrokeWidth}px`)
                    .style("cursor", "pointer")
                    .style("opacity", 1)
                    .raise(); // move to front
                d3.select(`#point2_${step}`)
                    .style("fill", color2)
                    .attr("d", triangleD3.size(24))
                    .style("stroke", "white")
                    .style("stroke-width", `${stepStrokeWidth}px`)
                    .style("cursor", "pointer")
                    .style("opacity", 1)
                    .raise(); // move to front
            }
        }

        resetStep = (step) => {
            step = `${step}`
            if (displaySteps.includes(step)) return;
            d3.select(`#pointfull_${step}`)
                .style("fill", "#61a3a9")
                .attr("d", circleD3.size(64))
                .style("stroke-width", `${stepStrokeWidth}px`)
                .style("opacity", stepOpacity);

            if (data2) d3.select(`#pointfull2_${step}`)
                .style("fill", "#a961a3")
                .attr("d", triangleD3.size(64))
                .style("stroke-width", `${stepStrokeWidth}px`)
                .style("opacity", stepOpacity);

            if (data1 && data2) {
                d3.select(`#point_${step}`)
                    .style("fill", "#61a3a9")
                    .attr("d", circleD3.size(24))
                    .style("stroke-width", `${stepStrokeWidth * 0.5}px`)
                    .style("opacity", stepOpacity);

                d3.select(`#point2_${step}`)
                    .style("fill", "#a961a3")
                    .attr("d", triangleD3.size(24))
                    .style("stroke-width", `${stepStrokeWidth * 0.5}px`)
                    .style("opacity", stepOpacity);
            }
        }

        charts.filter((c) => c.sync).forEach((c) => {
            if (c.step && c.color) highlightStep(c.step, c.color)
        })

        let defaultStep = $$('value').value
        const color = (isMultiMode()) ? "#FFA500" : "#ff00ed"
        const color2 = (isMultiMode()) ? "#1C64F2" : "#ff00ed"
        highlightStep(defaultStep, color, color2)
        hoveredStep.push(defaultStep);
    }
}

const lineChange = (slow_mode = false) => {
    if (!isConfirmed()) return;
    const value = $$('range').value;
    let lValue = 999 - value;
    $$('value').value = lValue;

    // implement step change, only allowed step can be selected, or the nearest step will be selected
    if (samplingSteps) {
        const nearestStep = sampledSteps.reduce((a, b) => Math.abs(b - lValue) < Math.abs(a - lValue) ? b : a);
        if (nearestStep !== lValue) {
            $$('range').value = 999 - nearestStep;
            $$('value').value = nearestStep;
            lValue = nearestStep;
        }
    }

    if (lineChangeInterval) {
        clearInterval(lineChangeInterval);
        lineChangeInterval = null;
    }

    if (hoveredStep) {
        hoveredStep.forEach(s => resetStep(s));
        hoveredStep = []
    }
    if (hoverStep) {
        hoverStep(lValue);
        hoveredStep.push(lValue);
    }

    if ($$('titlepreview')) $$('titlepreview').textContent = 'Step: ' + lValue
    if ($$('titlepreview2')) $$('titlepreview2').textContent = 'Step: ' + lValue

    if (slow_mode === true) {
        slow_mode_count += 1;
        if (slow_mode_count < 4) {
            console.log('slow mode')
            return;
        } else {
            slow_mode_count = 0;
        }
    }

    if (Date.now() - lastDownload < 100) {
        console.log('too fast move, restart in 1 s')
        lineChangeInterval = setInterval(() => lineChange(), 1000)
        return
    }
    lastDownload = Date.now();

    updatePreview(lValue)
}

const switchPreview = (multi = true) => {
    const $melpreview = $$('preview_container');
    const $melpreview2 = $$('preview_container2');
    if (multi) {
        $melpreview.classList.replace('w-[700px]', 'w-[320px]')
        $melpreview2.classList.remove('hidden')
    } else {
        $melpreview.classList.replace('w-[320px]', 'w-[700px]')
        $melpreview2.classList.add('hidden')
    }
}

const updatePreview = (sIndex = 999, reset = false) => {
    // TODO: update preview
    const color = (isMultiMode()) ? "#FFA500" : "#ff00ed"
    const color2 = (isMultiMode()) ? "#1C64F2" : "#ff00ed"

    let cards = []
    const width = isMultiMode() ? 320 : 700;
    const height = 200;

    const indexMode = config.pathData[currentMode].indexMode ?? "key";
    const previewCards = [
        {
            id: '', display: true,
            svg: circleD3,
            csvSrc: () => {
                if (indexMode === "key") {
                    return getCsvSrc(sIndex)
                }
                if (indexMode === "number") {
                    return getCsvSrc(sIndex, currentSinger[0], currentTargetSinger[0], findCorrespondingSong(currentSinger[0]))
                }
            },
            title: `Step: ${sIndex}`,
            titleColor: color,
            label: () => isMultiMode() ? `${mapToSongFunc(currentSong[0])}: ${mapToNameFunc(currentSinger[0])} -> ${mapToNameFunc(currentTargetSinger[0])}` : ''
        },
        {
            id: '2', display: isMultiMode(), svg: triangleD3,
            csvSrc: () => {
                if (indexMode === "key") {
                    return getCsvSrc(sIndex, currentSinger[currentSinger.length - 1], currentTargetSinger[currentTargetSinger.length - 1], currentSong[currentSong.length - 1])
                }
                if (indexMode === "number") {
                    return getCsvSrc(sIndex, currentSinger[currentSinger.length - 1], currentTargetSinger[currentTargetSinger.length - 1], findCorrespondingSong(currentSinger[currentSinger.length - 1]))
                }
            },
            title: `Step: ${sIndex}`,
            titleColor: color2,
            label: () => `${mapToSongFunc(currentSong[currentSong.length - 1])}: ${mapToNameFunc(currentSinger[currentSinger.length - 1])} -> ${mapToNameFunc(currentTargetSinger[currentTargetSinger.length - 1])}`
        },
    ]

    previewCards.forEach((card) => {
        if (!card.display) return;

        const { id, csvSrc, title, titleColor, label } = card;
        // generate div
        const refId = `preview${id}`;

        if (!reset) {
            // update audio
            if (changeVideoTimer[refId]) clearTimeout(changeVideoTimer[refId])
            changeVideoTimer[refId] = setTimeout(() => {
                $$(`video${refId}`).src = csvSrc().replace('.csv', '.wav')
            }, 300);

            // update graph
            d3.csv(csvSrc(), (error, data) => {
                if (error) { console.error(error); return; }
                const arrayData = data.map(row => Object.values(row).map(d => +d));
                charts = charts.filter((c) => c.id !== refId)
                // draw mel spectrogram
                plotMelSpectrogram(arrayData, refId, titleColor, close, width, height, true, hidePitch);
            });
            return;
        }

        const div = getDiv(refId, csvSrc().replace('.csv', '.wav'), titleColor, title, label(), !!card.svg, false);
        const $container = $$(`preview_container${id}`);
        if ($container) {
            $container.innerText = "";
            $container.appendChild(div);
        }

        // get data and bind div
        d3.csv(csvSrc(), (error, data) => {
            if (error) console.error(error);
            bindDiv(refId, data, card.svg ?? null, titleColor, () => { }, width, height)
        });
        cards.push(div)
    });

    switchPreview(isMultiMode())
}

const plotMelSpectrogram = (melData, refId, title_color, close, width = 345, height = 200, sync = true, noPitch = false) => {
    const getColorMSE = (color1, color2) => {
        const c1 = d3.rgb(color1);
        const c2 = d3.rgb(color2);
        return Math.sqrt(d3.mean([c1.r - c2.r, c1.g - c2.g, c1.b - c2.b].map(d => d * d)))
    }

    const getDeltaEColor = (step) => {
        let r = 0.0;
        let g = 0.0;
        let b = 0.0;

        if (step <= 5) {
            step = 0;
        }

        if (step > 255) {
            step = 255;
        }

        if (step <= 10) {
            return [r, g, b];
        } else if (step <= 11) {
            r = 0.75 - (step - 5) * 0.75 / 6.0;
            g = 0.375 - (step - 5) * 0.375 / 6.0;
            b = 1.0;
        } else if (step <= 19) {
            g = (step - 11) / 8.0;
            b = 1.0;
        } else if (step <= 27) {
            g = 1.0;
            b = 1.0 - (step - 19) / 8.0;
        } else if (step <= 37) {
            r = (step - 27) / 10.0;
            g = 1.0;
        } else if (step <= 47) {
            r = 1.0;
            g = 1.0 - (step - 37) * 0.5 / 10.0;
        } else if (step <= 255) {
            r = 1.0;
            g = 0.5 - (step - 47) * 0.5 / 208.0;
        }

        return [r * 255, g * 255, b * 255];
    }

    // melData struct:
    // [
    //     [0.1, 0.2, 0.3, ...(x axis)],
    //     [0.1, 0.2, 0.3, ...],
    //     ...(y axis)
    // ]
    // compareMode: not draw pitch line

    let melData2 = null;

    if (melData.length === 2) {
        [melData, melData2] = melData;
    }

    // init
    const mmargin = { mtop: 5, mright: 40, mbottom: 35, mleft: 35 };
    const mwidth = width - mmargin.mleft - mmargin.mright;
    const mheight = height - mmargin.mtop - mmargin.mbottom;
    let start = 0, end = melData[1].length, start_index = 0

    // Create a Canvas element and append it to the container
    const canvas = document.createElement('canvas');
    canvas.width = (mwidth + mmargin.mleft + mmargin.mright);
    canvas.height = (mheight + mmargin.mtop + mmargin.mbottom);
    const context = canvas.getContext('2d');
    document.querySelector(`#mel${refId}`).innerHTML = ""
    document.querySelector(`#mel${refId}`).appendChild(canvas);

    let x = d3.scaleLinear().range([mmargin.mleft, mmargin.mleft + mwidth]).domain([0, melData[1].length]);
    let y = d3.scaleLinear().range([mheight + mmargin.mtop, mmargin.mtop]).domain([0, melData.length - 1]);

    let color, color2, color_lock;

    if (!color_lock) {
        const min = d3.min(melData.filter((_, i) => i !== 0), array => d3.min(array))
        const max = d3.max(melData.filter((_, i) => i !== 0), array => d3.max(array))

        color = d3.scaleSequential(d3.interpolateViridis).domain([min, max])

        if (melData2) {
            const min2 = d3.min(melData2.filter((_, i) => i !== 0), array => d3.min(array))
            const max2 = d3.max(melData2.filter((_, i) => i !== 0), array => d3.max(array))
            color2 = d3.scaleSequential(d3.interpolateViridis).domain([min2, max2])
            color_lock = true
        }
    }

    const pitchYMax = 600; //d3.max(melData[0]); // TODO: make it to (step 0's top) * 1.5
    const pitchDataY = d3.scaleLinear().range([mheight + mmargin.mtop, mmargin.mtop]).domain([0, pitchYMax]);

    // Draw the rectangles on the Canvas
    const drawRectangles = () => {
        let w = Math.ceil(mwidth / melData[0].length);
        if (w < 2) w = 2; // Make sure the rectangles are visible
        let h = Math.ceil(mheight / melData.length);

        // reset x axis after zoom
        x = d3.scaleLinear().range([mmargin.mleft, mmargin.mleft + mwidth]).domain([0, end - start]);

        // Draw the rectangles
        for (let i = 1; i < melData.length; i++) {
            for (let j = 0; j < melData[i].length; j++) {
                if (!showFrequency) {
                    continue
                }
                if (showFrequency.length == 2) {
                    [lower, upper] = showFrequency
                    if (i > upper) continue;
                    if (i < lower) continue;
                }

                if (melData2) {
                    // calculate the difference using deltaE
                    const color_1 = color(melData[i][j]);
                    const color_2 = color2(melData2[i][j]);
                    const deltaE = getColorMSE(color_1, color_2);
                    const rgb = getDeltaEColor(deltaE);
                    context.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
                } else {
                    context.fillStyle = color(melData[i][j]);
                }
                context.fillRect(x(j), y(i - 1) - h, w, h);
            }
        }

        if (!noPitch && !hidePitch) {
            // Draw pitch lines
            context.fillStyle = 'red';
            context.strokeStyle = 'red';
            context.lineWidth = 1;

            // get data from melData[0]
            const pitchData = melData[0];
            const pitchDataLength = pitchData.length;

            for (let i = 0; i < pitchDataLength; i++) {
                const pitch = pitchData[i];
                if (pitch === 0) continue; // skip 0 pitch
                context.fillRect(x(i), pitchDataY(pitch), w, h);
            }
        }

        // Color for the axis and labels
        context.fillStyle = darkMode ? 'white' : 'black';
        context.strokeStyle = darkMode ? 'white' : 'black';

        // Draw X-label text
        context.font = '12px Times New Roman';
        context.fillText('Time (s)', canvas.width / 2 - 20, mheight + mmargin.mtop + mmargin.mbottom - 5);

        if (showFrequency) {
            // Draw Y-label text
            context.save();
            context.rotate(-Math.PI / 2);
            context.font = '12px Times New Roman';
            context.textAlign = 'center';
            context.fillText('Channel', - canvas.height / 2 + 15, mmargin.mleft - 25);
            context.restore();
        }

        const drawLine = (x1, y1, x2, y2) => {
            context.beginPath();
            context.moveTo(x1, y1);
            context.lineTo(x2, y2);
            context.stroke();
        }

        // Draw X-axis line
        drawLine(mmargin.mleft, mheight + mmargin.mtop, mwidth + mmargin.mleft, mheight + mmargin.mtop);
        if (showFrequency) {
            // Draw Y-axis line
            drawLine(mmargin.mleft, mmargin.mtop, mmargin.mleft, mheight + mmargin.mtop);
        }

        if (!noPitch && !hidePitch) {
            // Draw Y-label 2
            context.save();
            context.fillStyle = 'red';
            context.strokeStyle = 'red';

            context.textAlign = 'center';
            context.rotate(Math.PI / 2)
            context.fillText('F0 (Hz)', canvas.height / 2 - 15, - canvas.width + 10);
            context.restore();

            context.save();
            context.fillStyle = 'red';
            context.strokeStyle = 'red';
            drawLine(mmargin.mleft + mwidth + 1, mmargin.mtop, mmargin.mleft + mwidth + 1, mheight + mmargin.mtop);
            context.restore();
        }

        // Add X-axis labels
        const step = Math.round(melData[1].length / 5);
        const drawStep = [];
        for (let i = 0; i < melData[1].length; i++) {
            const text = (i + start_index) * 256 / 24000;
            if (start_index === 0) {
                if (text.toString().length > 3) continue;
            } else {
                if (i % step !== 0) continue;
            }
            drawStep.push(i);
        }
        const newDrawStep = [];
        if (drawStep.length > 5) {
            const step = Math.round(drawStep.length / 5);
            for (let i = 0; i < drawStep.length; i++) {
                if (i % step === 0) {
                    newDrawStep.push(drawStep[i]);
                }
            }
        } else {
            newDrawStep.push(...drawStep);
        }
        for (let i = 0; i < newDrawStep.length; i++) {
            const text = (newDrawStep[i] + start_index) * 256 / 24000;
            const xPos = x(newDrawStep[i]);
            context.fillText(text.toFixed(2), xPos - 10, mheight + mmargin.mtop + 15);
            drawLine(xPos, mheight + mmargin.mtop, xPos, mheight + mmargin.mtop + 5);
        }

        // Add Y-axis labels
        if (showFrequency) {
            for (let i = 0; i <= melData.length; i++) {
                if (i % 20 !== 0) continue;
                const yPos = y(i);
                const yOff = i >= 100 ? -15 : i >= 10 ? -10 : -5;
                context.fillText(i, mmargin.mleft - 10 + yOff, yPos + 5);
                drawLine(mmargin.mleft - 5, yPos, mmargin.mleft, yPos)
            }
        }
        if (!noPitch && !hidePitch) {
            context.save();
            context.fillStyle = 'red';
            context.strokeStyle = 'red';
            // Add Y-axis labels
            for (let i = 0; i <= pitchYMax; i++) {
                if (i % 100 !== 0) continue;
                const yPos = pitchDataY(i);
                context.fillText(i, mmargin.mleft + mwidth + 5, yPos + 5);
                drawLine(mmargin.mleft + mwidth, yPos, mmargin.mleft + mwidth + 5, yPos)
            }
            context.restore();
        }

    }

    // Implement brushing manually
    let isBrushing = false;
    let startX, endX, lastX;

    // Initial drawing
    drawRectangles();

    // Initial zooming
    canvas.addEventListener('mousedown', (e) => {
        isBrushing = true;
        startX = e.clientX - canvas.getBoundingClientRect().left;
        lastX = startX;
    });
    canvas.addEventListener('mousemove', (e) => {
        if (isBrushing) {
            endX = e.clientX - canvas.getBoundingClientRect().left;
            // Draw the brush selection
            if (startX < endX) {
                // go right
                if (lastX > endX) {
                    clear();
                    drawRectangles();
                    context.fillStyle = 'rgba(0, 0, 0, 0.4)';
                    context.fillRect(startX, 0, endX - startX, canvas.height);
                } else {
                    context.fillStyle = 'rgba(0, 0, 0, 0.4)';
                    context.fillRect(lastX, 0, endX - lastX, canvas.height);
                }
            } else if (lastX < endX) {
                clear();
                drawRectangles();
                context.fillStyle = 'rgba(0, 0, 0, 0.4)';
                context.fillRect(endX, 0, startX - endX, canvas.height);
            } else {
                context.fillStyle = 'rgba(0, 0, 0, 0.4)';
                context.fillRect(endX, 0, lastX - endX, canvas.height);
            }
            lastX = endX;
        }
    });
    canvas.addEventListener('mouseup', () => {
        isBrushing = false;
        if (!startX) return
        if (!endX) [startX, endX] = [startX - 30, startX + 30]
        // Reset the canvas to the original state
        clear()
        drawRectangles();
        // Handle the selected range (startX to endX)
        // Clip the data to the selected range
        let start = Math.round(x.invert(startX));
        let end = Math.round(x.invert(endX));
        if (start === end) return;
        if (Math.abs(start - end) < 5) {
            console.log('Zoomed in too much!')
            return
        }
        // convert start and end into percentage
        start = start / melData[0].length;
        end = end / melData[0].length;
        if (sync) {
            charts.forEach(c => c.sync && c.zoom(start, end))
            zoomStart = start
            zoomEnd = end
        } else {
            zoom(start, end)
        }
    });

    // Reset the mel spectrogram when a new song is selected
    const reset = () => {
        let chart = charts.find(c => c.id === refId)
        if (!chart) {
            console.log('Reset fail: not find id', refId)
            return;
        }
        if (chart.sync) {
            zoomStart = 0;
            zoomEnd = 0;
        }
        startX = 0;
        endX = 0;
        lastX = 0;
        isBrushing = false;
        melData = chart.melData;
        if (chart.melData2) melData2 = chart.melData2;
        start = 0;
        end = melData[0].length;
        start_index = 0;
        clear();
        drawRectangles();
    }

    const clear = () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
    }

    const zoom = (_start, _end) => {
        if (!_start || !_end) return;
        if (_start > _end) [_end, _start] = [_start, _end];
        if (_start < 0) _start = 0;
        // convert start and end percentage into index
        _start = Math.round(_start * melData[0].length);
        _end = Math.round(_end * melData[0].length);
        if (_end - _start < 5) {
            return
        }
        start = _start
        end = _end
        melData = melData.map(row => row.slice(_start, _end));
        if (melData2) {
            melData2 = melData2.map(row => row.slice(_start, _end));
        }
        start_index += start;
        clear();
        drawRectangles();
    }

    const highlight = (time) => {
        const xPos = x(Math.round(time * 24000 / 256));
        if (xPos < 0 || xPos > canvas.width - mmargin.mright) return;
        context.fillStyle = 'red';
        context.fillRect(xPos, mmargin.mtop, 1.5, canvas.height - mmargin.mtop - mmargin.mbottom);
    }

    const chart = {
        id: refId,
        step: Number(refId.split('_')[1]),
        sync: sync,
        selected: false,
        canvas: canvas,
        x: x,
        y: y,
        color: title_color,
        melData: melData,
        melData2: melData2 ?? null,
        reset: reset,
        zoom: zoom,
        highlight: highlight,
        close: close
    };

    charts.push(chart);
}

const drawCurve = (id, width, height) => {
    const singer = currentSinger[0];
    const song = currentSong[0];
    const target = currentTargetSinger[0];
    const csvSrc = getMetricsSrc(singer, target, song);
    d3.csv(csvSrc, (error, data) => {
        if (error) { console.error(error); return; }
        const $container = $$(`metrics${id}`);

        const marginTop = 10;
        const marginRight = 50;
        const marginBottom = 30;
        const marginLeft = 40;

        const keysWithLable = [
            "Dembed (↑)",
            "F0CORR (↑)",
            "FAD (↓)",
            "F0RMSE (↓)",
            "MCD (↓)",
        ];

        const keys = keysWithLable.map((k) => k.split(' ')[0]);
        const values = keys.map((key) => data.map((d) => {
            const v = +d[key];
            // const step = d.step;
            if (isNaN(v)) return null;
            // if (v < 0) return 0.1;
            return autoLog(v);
        }));

        const color = d3.scaleOrdinal()
            .domain(keysWithLable)
            .range(metricsColors);

        const $swatch = Swatches(color, { columns: "100px", marginLeft: 10 });

        $container.appendChild($swatch);

        const squeezeArray = (array) => {
            let result = [];
            for (let i = 0; i < array.length; i++) {
                result = result.concat(array[i]);
            }
            return result;
        };

        const squeezedValues = squeezeArray(values);

        const x = d3.scaleLog()
            .clamp(true).domain([0.5, d3.max(data, (d) => +d.step)])
            .rangeRound([width - marginRight, marginLeft])
            .base(2).nice();

        const y = d3.scaleLinear()
            .domain([0, d3.max(squeezedValues)])
            .rangeRound([height - marginBottom, marginTop]);

        const line = d3.line()
            .defined((y, i) => !isNaN(data[i].step) && !isNaN(y) && x(data[i].step) !== Infinity)
            .x((d, i) => x(data[i].step))
            .y(y);

        // Function to find the closest point to the mouse
        const findClosestStep = (mouseX) => {
            let minDist = Infinity;
            let findClosestStep = null;

            data.forEach((d, i) => {
                const xVal = x(+d.step);
                const dist = Math.abs(mouseX - xVal);
                if (dist < minDist) {
                    minDist = dist;
                    findClosestStep = { x: xVal, step: d.step };
                }
            });

            return findClosestStep;
        }

        const svg = d3.select($container).append("svg")
            .attr("style", `color: ${darkMode ? 'white' : 'black'};`)
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [0, 0, width, height])
            .attr("style", "max-width: 100%; height: auto;");

        svg.append("g")
            .attr("transform", `translate(0,${height - marginBottom})`)
            .call(d3.axisBottom(x).ticks(width / 80).tickFormat((d) => d >= 1 ? (d === 1024 ? 1000 : d) : 0))
            .call((g) => g.select(".domain").attr("stroke", "currentColor"))
            .call((g) => g.append("text")
                .attr("x", width / 2)
                .attr("y", marginBottom - 4)
                .attr("fill", "currentColor")
                .attr("text-anchor", "center")
                .text("Step (log)"));

        const yAxis = svg.append("g")
            .attr("transform", `translate(${marginLeft},0)`)
            .call(d3.axisLeft(y).tickValues(d3.ticks(0.01, d3.max(squeezedValues), 10)).tickFormat((d) => d > 1 ? Math.pow(4, d).toFixed(0) : d.toFixed(1)))
            .call((g) => g.select(".domain").attr("stroke", "currentColor"))
            .call(g => g.append("text")
                .attr("x", -height / 2 + 30)
                .attr("y", -marginLeft + 10)
                .attr("fill", "currentColor")
                .attr("transform", "rotate(-90)")
                .text("Score"))
            .call(g => g.selectAll(".tick line").clone()
                .attr("x2", width - marginLeft - marginRight)
                .attr("stroke-opacity", 0.1))

        svg.append("g")
            .attr("fill", "none")
            .attr("stroke-width", 1.5)
            .attr("stroke-linejoin", "round")
            .attr("stroke-linecap", "round")
            .selectAll()
            .data(values)
            .enter()
            .append("path")
            .attr("stroke", (d, i) => metricsColors[i])
            .attr("d", line);

        svg.selectAll("text").attr("fill", "currentColor")
        svg.selectAll("line").attr("stroke", "currentColor")

        // Append a group element for the dot
        const dotGroup = svg.append("g")
            .style("display", "none");

        dotGroup
            .append("g")
            .selectAll()
            .data(keys)
            .enter()
            .append("circle")
            .attr("r", 4)
            .attr("id", (d) => `dot_${d}`)
            .attr("fill", (d, i) => metricsColors[i]);

        dotGroup
            .append("g")
            .selectAll()
            .data(keys)
            .enter()
            .append("text")
            .attr("id", (d) => `dot_text_${d}`)
            .attr("fill", (d, i) => metricsColors[i])
            .attr("font-size", 11)
            .attr("font-weight", 600)
            .attr("text-anchor", "start")
            .attr("stroke", "white")
            .attr("stroke-width", 3)
            .attr("font-family", "sans-serif")
            .attr("paint-order", "stroke")
            // .attr("alignment-baseline", "center")
            .text((d) => d);

        // Mouse move event listener
        svg.on("mousemove", function (event) {
            const [mouseX, mouseY] = d3.mouse(this)
            const closestStep = findClosestStep(mouseX);

            let yVals = [];

            if (closestStep) {
                // add a debunce to avoid too many update
                $range.value = 999 - closestStep.step;
                lineChange(false);

                dotGroup.style("display", null);
                keys.forEach((key, i) => {
                    const yVal = +data.find((d) => d.step === closestStep.step)[key];

                    const drawYVal = y(autoLog(yVal));
                    yVals.push({
                        key: key,
                        y: drawYVal,
                        text: yVal.toFixed(2),
                        difference: Math.abs(drawYVal - mouseY),
                    });

                    d3.select(`#dot_${key}`)
                        .attr("cx", closestStep.x)
                        .attr("cy", drawYVal)
                        .raise();
                });
            }

            // Draw the number (only the closest step)
            yVals = yVals.sort((a, b) => a.difference - b.difference);

            yVals.forEach((v, i) => {
                if (i === 0) {
                    d3.select(`#dot_text_${v.key}`)
                        .attr("x", closestStep.x + 10)
                        .attr("y", v.y + 3)
                        .text(v.text)
                        .raise();
                } else {
                    d3.select(`#dot_text_${v.key}`).attr("x", -100).attr("y", -100);
                }
            });
        });

        // Mouse out event listener
        // svg.on("mouseout", function () {
        //     dotGroup.style("display", "none");
        // });

        return svg.node();
    });
}

// ==== UI interaction functions ====
const resetDisplay = (clear = true, replot = true, keep_stepmap = false) => {
    // update choosed options
    updateSelect("mode_id", availableMode, currentMode);
    const pic_index = config.picTypes.indexOf(currentShowingPic) ?? 3;
    if (userMode == "basic") {
        initSelect("pic_id", config.picTypes, pic_index, 4);
    } else {
        initSelect("pic_id", config.picTypes, pic_index, -1);
    }
    // updateSelect("pic_id", , );
    updateOptions("sourcesinger_id", currentSinger);
    updateOptions("target_id", currentTargetSinger);
    updateOptions("song_id", currentSong);
    if (!isConfirmed()) return;
    if (clear) {
        $$("mel_card_container").innerHTML = "";
        $$("tips").style = 'display: block';
        displaySteps = [];
        charts = [];
        usedColorList = [];
    }

    if (userMode === "basic") {
        if (currentMode === "Step Comparison" && !$$("grid_table")) loadGrid()
        $$("add_preview").classList.add("hidden");
    } else {
        $$("add_preview").classList.remove("hidden");
    }

    if (isSelectable()) {
        if (displaySteps.length == 0) {
            selectStep(-1)
            console.log('Single step mode init')
        }
        $$("add_preview").classList.add("hidden");
    } else {
        $$("add_preview").classList.remove("hidden");
    }

    if (keep_stepmap) return;
    $$("dataviz_axisZoom").innerHTML = "";
    if (currentShowingPic === "encoded_step") {
        drawStepMap(`${baseLink}/data/mp_all/step_encoder_output.csv`);
    } else if (isMultiMode()) {
        const indexMode = (config.pathData[currentMode].indexMode ?? "key") === "number";

        drawStepMap(
            getStepSrc(
                currentSinger[0],
                currentTargetSinger[0],
                indexMode ? findCorrespondingSong(currentSinger[0]) : currentSong[0]
            ),
            getStepSrc(
                currentSinger[currentSinger.length - 1],
                currentTargetSinger[currentTargetSinger.length - 1],
                indexMode ? findCorrespondingSong(currentSinger[currentSinger.length - 1]) : currentSong[currentSong.length - 1]
            ));
    } else {
        drawStepMap(getStepSrc());
    }
    if (isMultiMode()) {
        $$("control_panel").classList.remove("gap-0.5");
        $$("control_panel").classList.add("gap-1", "my-1");
    } else {
        $$("control_panel").classList.remove("gap-1", "my-1");
        $$("control_panel").classList.add("gap-0.5");
    }
    updateHistogram();


    $$("dataviz_axisZoom").addEventListener("mouseleave", () => {
        const step = 999 - $range.value;
        hoverStep(step)
        hoveredStep.push(step);
    })

    $$("dataviz_axisZoom").addEventListener("mouseenter", () => {
        if (hoveredStep) {
            hoveredStep.forEach(step => resetStep(step));
            hoveredStep = []
        }
    })

    if (!replot) return;
    $$("range").value = 0;
    $$("value").value = 999;

    updatePreview(999, true);
}

const showBestCase = () => {
    // find the best evaluation data by currentHistogram
    const selected = currentHistogram[0]["name"] ?? "";
    if (selected === "") return;
    const bestCase = config.evaluation_data.find((d) => d.best.includes(selected));
    // print the best one
    currentMode = "Metric Comparison";
    refreshOptions(true, bestCase.sourcesinger);
    currentSinger = [bestCase.sourcesinger];
    currentSong = [bestCase.song];
    currentTargetSinger = [bestCase.target];
    lockOptions();
    resetDisplay();
}

const selectStep = (sIndex) => {
    const width = 290;
    const height = 200;
    if (!isConfirmed()) {
        alert('Please select data first')
        return;
    }
    sIndex = `${sIndex}`; // ! conver to string
    // if (displaySteps.length === 0) { // clear the pin area
    //     $$("mel_card_container").innerHTML = "";
    //     $$("tips").style = 'display: none'
    // }
    if (sIndex !== "-1" && displaySteps.indexOf(sIndex) !== -1) {
        charts.filter(c => c.id.endsWith(sIndex))?.forEach(c => c.close()); // close the card if the func called again
        return
    }
    if (`${sIndex}`.indexOf('ph') !== -1) { // if it is placeholder
        charts.filter(c => c.id.endsWith(sIndex))?.forEach(c => c.close()); // remove the placeholder if exist

        // generate a placeholder
        const div = document.createElement('div');
        div.className = 'mel_card';
        div.id = `mel_${sIndex}`;
        div.style.width = `${width}px`;
        div.style.height = `${height}px`;
        if (currentMode === 'Step Comparison' && userMode === 'basic') {
            $mel.insertBefore(div, $mel.lastChild);
        } else {
            $mel.appendChild(div);
        }
        charts.push({ // add a close function
            id: `_${sIndex}`, close: () => {
                charts = charts.filter(c => c.id !== `_${sIndex}`);
                displaySteps = displaySteps.filter(d => d !== sIndex)
                div.remove()
            }
        });
        displaySteps.push(sIndex) // add to displaySteps
        return
    }
    if (displaySteps.length >= 3 && currentMode === "Step Comparison" && userMode === "basic") {
        // up to 2 is allowed in basic mode
        // other will be ignored
        console.log('basic mode only allows 2 steps')
        addComparison(sIndex)
        return
    }
    if (displaySteps.length >= 3 || (isSelectable() && sIndex !== "-1")) {
        // when slots full, only 3 cards can be displayed: more selected will be seen as hovered
        // for unselectable mode, none card will be displayed, only hovered
        // change hovered
        $$('range').value = 999 - sIndex;
        lineChange();
        return
    }
    let color;
    displaySteps.push(sIndex)
    if (!isNaN(parseInt(sIndex)) && parseInt(sIndex) >= 0) {
        color = config.colorList.map(c => c).filter(c => !usedColorList.includes(c))[0] ?? "#000";
        usedColorList.push(color);
        console.log('color', color)
        highlightStep(sIndex, color) // color needed
    } else {
        color = "#000"
    }

    let cards = []

    if (!isMultiMode() && !(currentMode === "Step Comparison" && userMode === "basic")) {
        // Function to handle the mouse down event
        const dragStart = (e) => {
            currentCard = e.target
            setTimeout(() => {
                currentCard.classList.add('border-dashed', 'border-2', 'border-blue-500')
            })
        }

        setTimeout(() => {
            const card = cards[0];
            if (!card) return;
            card.addEventListener('dragstart', dragStart);

            const $mel_canvas = $$(`mel_${sIndex}`);
            $mel_canvas.addEventListener('mouseenter', () => {
                card.setAttribute('draggable', false)
            });
            $mel_canvas.addEventListener('mouseout', () => {
                card.setAttribute('draggable', true)
            });
        })
    }

    const close = (skip = false) => {
        if (isMultiMode()) {
            alert('Card in multi mode cannot be closed.')
            return;
        }
        if (currentMode === 'Metric Comparison') {
            alert('Card in Metric Comparison mode cannot be closed.')
            return;
        }
        // color set to default
        if (currentMode === 'Step Comparison' && userMode === 'basic') {
            gridComparison = gridComparison.filter((d) => d !== sIndex)
            if (!skip) {
                if ($$("mel_ph1")) {
                    gridComparison.push("ph2")
                    selectStep("ph2")
                } else {
                    gridComparison.push("ph1")
                    selectStep("ph1")
                }
            }
        }

        displaySteps = displaySteps.filter((d) => d !== sIndex)
        usedColorList = usedColorList.filter(c => c !== color);
        resetStep(sIndex)

        charts = charts.filter(c => c.id !== `_${sIndex}` && c.id !== `2_${sIndex}`);
        cards.forEach((c) => c.remove())
    }

    const [types] = getMultipleLable();

    const indexMode = config.pathData[currentMode].indexMode ?? "key";
    let index
    if (indexMode === "number") {
        // get index from config.pathData, then get corresponding song name
        const singer = currentSinger[currentSinger.length - 1];
        const song = currentSong[0];
        const songs = config.pathData[currentMode].data.map((d) => d.pathMap[singer]?.songs).flat().filter((s) => s !== undefined);
        index = songs.indexOf(song)
    }

    const referenceCards = [
        {
            id: '3', display: isMultiMode() && enableReference,
            svg: isMultiMode() && enableReference && ((types == 'song' || types == 'sourcesinger') ? circleD3 : null),
            color: isMultiMode() && enableReference && ((types == 'song' || types == 'sourcesinger') ? "#FFA500" : (darkMode ? "#fff" : "#000")),
            csvSrc: () => {
                if (indexMode === "key") {
                    return getReferenceCsvSrc(currentSinger[0], currentSong[0])
                }
                if (indexMode === "number") {
                    const correspondingSong = config.pathData[currentMode].data.find((d) => Object.keys(d.pathMap).includes(currentSinger[0])).pathMap[currentSinger[0]].songs[index]
                    return getReferenceCsvSrc(currentSinger[0], correspondingSong)
                }
            },
            title: () => 'Source',
            label: () => `${mapToSongFunc(currentSong[0])}: ${mapToNameFunc(currentSinger[0])}`
        },
        {
            id: '4', display: isMultiMode() && enableReference && (types == 'song' || types == 'sourcesinger'),
            svg: triangleD3,
            color: "#1C64F2",
            csvSrc: () => {
                if (indexMode === "key") {
                    return getReferenceCsvSrc(currentSinger[currentSinger.length - 1], currentSong[currentSong.length - 1])
                }
                if (indexMode === "number") {
                    const correspondingSong = config.pathData[currentMode].data.find((d) => Object.keys(d.pathMap).includes(currentSinger[currentSinger.length - 1])).pathMap[currentSinger[currentSinger.length - 1]].songs[index]
                    return getReferenceCsvSrc(currentSinger[currentSinger.length - 1], correspondingSong)
                }
            },
            title: () => 'Source',
            label: () => `${mapToSongFunc(currentSong[currentSong.length - 1])}: ${mapToNameFunc(currentSinger[currentSinger.length - 1])}`
        },
        {
            id: '5', display: isMultiMode() && enableReference,
            svg: isMultiMode() && enableReference && ((types == 'target') ? circleD3 : null),
            color: isMultiMode() && enableReference && ((types == 'target') ? "#FFA500" : (darkMode ? "#fff" : "#000")),
            csvSrc: () => getTargetReferenceCsvSrc(currentTargetSinger[0], currentSong[0]),
            title: () => 'Target',
            label: () => mapToNameFunc(currentTargetSinger[0])
        },
        {
            id: '6', display: isMultiMode() && enableReference && (types == 'target'),
            svg: triangleD3,
            color: "#1C64F2",
            csvSrc: () => getTargetReferenceCsvSrc(currentTargetSinger[currentTargetSinger.length - 1], currentSong[currentSong.length - 1]),
            title: () => 'Target',
            label: () => mapToNameFunc(currentTargetSinger[currentTargetSinger.length - 1])
        },
        {
            id: '', display: !isMultiMode() && currentMode !== 'Metric Comparison' && sIndex !== "-1",
            color: color,
            svg: circleD3,
            csvSrc: () => getCsvSrc(sIndex),
            title: () => `Step: ${sIndex}`,
            label: () => ''
        },
        {
            id: '2', display: !isMultiMode() && currentMode === 'Metric Comparison',
            div: (refId) => {
                const div = document.createElement('div');
                div.innerHTML = `<div class="card p-2 w-full flex flex-col col-span-3 gap-1" id="display${refId}">` +
                    `<div class="flex items-center">` +
                    `<div class="flex flex-col ml-1 mr-1">` +
                    `<h5 class="text-base font-bold tracking-tight mb-0 text-[black] line-clamp-1 dark:text-[white]" id="title${refId}">Metric Curve over Diffusion Step</h5>` +
                    `</div>` +
                    `</div>` +
                    `<div class="mx-auto h-[250px] dark:text-[white]" id="metrics${refId}"></div>` +
                    `</div>`;
                return div.firstChild;
            },
        },
        {
            id: '1', display: currentMode === 'Step Comparison' && userMode === 'basic' && sIndex === "-1",
            div: (refId) => {
                const div = document.createElement('div');
                div.innerHTML = `<div class="card min-w-[305px] p-2 w-full flex flex-col gap-1" id="display${refId}">` +
                    `<div class="flex items-center">` +
                    `<div class="flex flex-col mx-auto">` +
                    `<h5 class="text-base font-bold tracking-tight mb-0 text-[black] line-clamp-1 dark:text-[white]" id="title${refId}">Step Comparison Matrix</h5>` +
                    `</div>` +
                    `</div>` +
                    `<div class="flex flex-row">` +
                    `<div class="mx-auto grow-0 flex justify-center items-center w-[250px] h-[250px] dark:text-[white]" id="grid_table"></div>` +
                    `<div class="grow-0 flex w-[50px]" id="grid_table_legend"></div>` +
                    `</div>` +
                    `</div>`;
                return div.firstChild;
            },
        }
    ];

    referenceCards.forEach((card) => {
        if (!card.display) return;

        const { id, div, csvSrc, title, label, color } = card;
        // generate div
        const refId = `${id}_${sIndex}`;
        if (div) {
            $mel.appendChild(div(refId));
            if (currentMode === 'Metric Comparison') {
                // metrics bind
                drawCurve(refId, 900, 220);
            }
            return;
        }
        const divContent = getDiv(refId, csvSrc().replace('.csv', '.wav'), color, title(), label(), !!card.svg);
        // insert the div into the last two slots in $mel
        if (currentMode === 'Step Comparison' && userMode === 'basic') {
            $mel.insertBefore(divContent, $mel.lastChild);
        } else {
            $mel.appendChild(divContent);
        }
        cards.push(divContent)
        // get data and bind div
        downloadingLock.push(refId)
        d3.csv(csvSrc(), (error, data) => {
            downloadingLock = downloadingLock.filter((d) => d !== refId)
            if (error) console.error(error);
            bindDiv(refId, data, card.svg ?? null, color, close, width, height)
        });
    });
}

const checkCompare = () => {
    // check if there are 2 or more charts selected
    // if so, pop a compare window showing the difference of mel spectrogram
    const selectedCharts = charts.filter((c) => c.selected);
    if (selectedCharts.length < 2) return;
    if (selectedCharts.length >= 3) {
        alert('Please select 2 steps to compare.');
        return;
    }
    selectedCharts.forEach((c) => {
        c.selected = false;
        $$(`select${c.id}`).checked = false;
    });
    const compareId = compareNum++;
    const selectedData = selectedCharts.map((c) => c.melData);

    // pop a window
    const div = document.createElement('div');
    div.innerHTML = `<div class="card flex flex-col absolute cursor-move z-10" id="compare${compareId}" draggable="true">` +
        `<div class="flex items-center">` +
        // `<input id="select${sIndex}" type="checkbox" value="" class="checkbox mb-2 mr-1">`+
        `<h5 class="card-title">Mel Spectrogram Difference</h5>` +
        `<a class="btn-sec ml-auto h-9 w-14" id="refreshcompare${compareId}">${refreshIcon}</a>` +
        `<a class="btn-sec ml-2 h-9 w-14" id="closecompare${compareId}">${closeIcon}</a>` +
        `</div>` +
        `<div class="flex flex-row">` +
        `<div class="mx-auto min-w-[355px]" id="melcompare${compareId}"></div>` +
        `<div class="w-[26px]"><img src="img/difference_bar.jpg"></div>` +
        `</div>` +
        `</div>`;
    const domNode = div.firstChild;
    $$('step_preview').appendChild(domNode)

    // move to center of screen
    domNode.style.left = `${(window.innerWidth - 345) / 2}px`;
    domNode.style.top = `${window.scrollY + (window.innerHeight - 200) / 2}px`;

    const $mel_canvas = $$(`melcompare${compareId}`);
    $mel_canvas.addEventListener('mouseenter', () => {
        domNode.setAttribute('draggable', false)
    });
    $mel_canvas.addEventListener('mouseout', () => {
        domNode.setAttribute('draggable', true)
    });

    let offsetX, offsetY, isDragging = false;

    const dragStart = (e) => {
        // let windows follow cursor
        e.preventDefault()
        isDragging = true;
        offsetX = e.clientX - domNode.getBoundingClientRect().left;
        offsetY = e.clientY - domNode.getBoundingClientRect().top;
        domNode.style.zIndex = 2; // Bring the element to the front
        domNode.style.cursor = "grabbing";
    }
    const drag = (e) => {
        if (isDragging) {
            const x = e.clientX - offsetX;
            const y = e.clientY - offsetY;

            // Ensure the draggable div stays within the viewport
            if (x >= 0 && x + domNode.offsetWidth <= window.innerWidth) {
                domNode.style.left = x + "px";
            }
            if (y >= 0 && y + domNode.offsetHeight <= window.innerHeight) {
                domNode.style.top = (window.scrollY + y) + "px";
            }
        }
    }
    const endDrag = () => {
        isDragging = false;
        domNode.style.zIndex = 1; // Restore the element's original z-index
        domNode.style.cursor = "move";
    }

    domNode.addEventListener('dragstart', dragStart);
    document.addEventListener("mousemove", drag);
    document.addEventListener("mouseup", endDrag);

    const close = () => {
        $$(`compare${compareId}`).remove()
    }

    $$(`closecompare${compareId}`).addEventListener('click', () => {
        console.log(`closecompare${compareId}`, 'clicked')
        close()
    })

    // plot the difference
    plotMelSpectrogram(selectedData, `compare${compareId}`, '#000', close, 345, 200, false, true);

    $$(`refreshcompare${compareId}`).addEventListener('click', () => {
        console.log(`refreshcompare${compareId}`, 'clicked')
        charts.find((c) => c.id === `compare${compareId}`)?.reset()
    })
}



const preloadingFile = (src) => {
    // preloading data files for faster loading
    const img = new Image();
    img.src = src;
    img.onload = () => {
        console.log('Preloaded:', src)
    }
    img.onerror = () => {
        console.log('Preloaded:', src)
    }
}
const preloading = () => {
    preloadingFile('img/difference_bar.jpg');
    // loading all steps
    // Array.from({ length: 1000 }, (_, i) => i).forEach((i) => {
    //     preloadingFile(getCsvSrc(i));
    //     preloadingFile(getCsvSrc(i).replace('.csv', '.wav'));
    // });
}