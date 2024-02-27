/**
 * Copyright (c) 2023 Amphion.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ==== Event Listeners ====
let titleClickCount = 0;
$$("title").addEventListener("click", (e) => {
    titleClickCount++;
    if (titleClickCount > 5) {
        titleClickCount = 0;
        const newTitle = prompt("Please enter new title", document.title);
        if (newTitle != null) {
            $$("title").innerHTML = newTitle;
            document.title = newTitle;
        }
    }
})

$$("mode_id").addEventListener("change", () => {
    const selectedValue = $$("mode_id").value;
    currentMode = selectedValue;
    if (currentMode !== "Metric Comparison") {
        if (currentHistogram.length > 0) d3.select(`#rect_${currentHistogram.shift().name}`).attr("stroke", "none")
        unlockOptions();
    }
    refreshOptions();
    resetDisplay();
});
$$("pic_id").addEventListener("change", () => {
    const selectedValue = $$("pic_id").value;
    currentShowingPic = selectedValue;
    resetDisplay(false, false);
});
$$("text_id").addEventListener("change", () => {
    const selectedValue = $$("text_id").checked;
    currentTextShow = selectedValue;
    resetDisplay(false, false);
});
$$("range").addEventListener("input", lineChange);
$$("reset_map").addEventListener("click", () => {
    resetDisplay(false, false);
})
$$("value").addEventListener("change", (e) => {
    const value = e.target.value;
    const lValue = 999 - value;
    $$('range').value = lValue;
    lineChange();
})

$$("step_axis").addEventListener('touchmove', (e) => {
    e.stopPropagation();
}, { passive: false });
// $$("reset_preview").addEventListener("click", () => {
//     charts.filter((c) => !c.sync).forEach((c) => c.reset());
// });
$$("add_preview").addEventListener("click", () => {
    if (displaySteps.length >= 3) {
        alert('Please remove step before pinning new one. Up to 3 steps can be pinned.')
        return
    }
    const step = 999 - $$("range").value;
    selectStep(step);
})
$$("help").addEventListener("click", () => {
    localStorage.removeItem('GUIDED');
    location.reload();
})
$$("metrics_help").addEventListener("click", () => {
    alert("Metric Tips", "Scores: The higher the better. Log scores: The lower the better. \n" +
        "The metrics are calculated by algorithm. Each metric is defined as follows:\n\n" +
        "- <b>Dembed (Singer Similarity)</b>: This quantitatively assesses the similarity between the timbre of the original singer's voice and the converted voice. It's calculated using the cosine similarity between feature vectors representing the timbre characteristics of the two voices. A higher similarity score indicates the more timbre similarity. \n" +
        "- <b>F0CORR (Pitch Correlation)</b>: This measures the Pearson Correlation Coefficient between the F0 values of the converted singing voice and the target voice. It assesses the linear relationship between the F0 contours of the two voices. A higher F0CORR indicates a stronger correlation and better F0 similarity. \n" +
        "- <b>FAD (Fréchet Audio Distance)</b>: This is a reference-free evaluation metric to evaluate the quality of audio samples. FAD correlates more closely with human perception. A lower FAD score indicates a higher quality of the audio. \n" +
        "- <b>F0RMSE (Pitch Accuracy)</b>: This measures the Root Mean Square Error of the Fundamental Frequency (F0) values between the converted singing voice and the target voice. It quantifies how accurately the F0 of the converted voice matches that of the target voice. A lower F0RMSE indicates better F0 accuracy. \n" +
        "- <b>Mel-cepstral distortion (MCD)</b>: This assesses the quality of the generated speech by comparing the discrepancy between generated and ground-truth singing voice. It measures how different the two sequences of mel cepstra are. A lower MCD indicates better quality.");
})
$$("projection_help").addEventListener("click", () => {
    alert("Projection Tips", "This view depicts the 128-dimensional diffusion steps in the diffusion model reduced to 2-dimensional embedding using t-SNE. Each point in the projection space represents a specific diffusion step. The arrangement of these points in 2D space illustrates how each step varies from others, providing perception of the progression and transformation occurring within the diffusion model.")
})
$$("control_help").addEventListener("click", () => {
    alert("Control Tips", "<b>Projection Embedding:</b> \n" +
        "- <b>Step (Diffusion Step)</b>: Output embedding of the diffusion step encoder, and it does not contain any other data. \n" +
        "- <b>Step + Noise</b>: Output embedding of the diffusion step encoder added with the initial random noise. \n" +
        "- <b>Step + Noise + Condition</b>: Output embedding of the diffusion step encoder added with the initial random noise and the model conditions, e.g., F0 and energy extracted from the source singing voice and speaker embedding corresponding to the target singer. \n" +
        "- <b>First/Middle/Last Layer</b>: The first, tenth, and final layer of the 20-layer residual blocks in the diffusion decoder. \n"
    )
})

$$("controls").addEventListener("click", () => {
    playMode = !playMode;
    updatePlayIcon(playMode)
    localStorage.setItem('AUTO_PLAY', playMode ? "true" : "false");
    if (parseInt($range.value) >= 999) {
        $range.value = 0;
    }
})

const preferColor = window.matchMedia('(prefers-color-scheme: dark)')
if (preferColor.matches) {
    darkMode = true;
} else {
    darkMode = false;
}
preferColor.addEventListener("change", (e) => {
    if (e.matches) {
        darkMode = true;
    } else {
        darkMode = false;
    }
    console.log('darkMode', darkMode);
    charts.forEach(c => c.reset())
    d3.select("#histogram")
        .attr("style", `color: ${darkMode ? 'white' : 'black'};`)
    d3.select("#histogram2")
        .attr("style", `color: ${darkMode ? 'white' : 'black'};`)
})

setInterval(() => {
    if (playMode) {
        // auto play
        $range.value = jumpStep + parseInt($range.value);
        if (parseInt($range.value) >= 999) {
            playMode = false
            updatePlayIcon(playMode)
        }
        lineChange(true);
    }
}, 50)

// ==== Drag and Drop ====
const drag = (e) => {
    e.preventDefault()
}
const dragEnter = (e) => {
    e.preventDefault()
    if (e.target.id.indexOf('display') === -1) return
    if (!currentCard) return

    let melArray = Array.from($mel.childNodes)
    let currentIndex = melArray.indexOf(currentCard)
    let targetindex = melArray.indexOf(e.target)

    if (currentIndex < targetindex) {
        $mel.insertBefore(currentCard, e.target.nextElementSibling)
    } else {
        $mel.insertBefore(currentCard, e.target)
    }
}
const dragEnd = (e) => {
    if (!currentCard) return;
    currentCard.classList.remove('border-dashed', 'border-2', 'border-blue-500')
    currentCard = null
}
document.addEventListener('dragover', drag);
document.addEventListener('dragenter', dragEnter);
document.addEventListener('dragend', dragEnd);