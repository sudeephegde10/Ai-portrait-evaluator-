/**
 * app.js — Main Application Logic
 *
 * Handles:
 * - Drag-and-drop and click-to-upload for both images
 * - Image preview display
 * - Fetch POST /analyze with FormData
 * - Render scores with animated circular progress bars
 * - Render radar chart via Chart.js
 * - Toggle heatmap overlay
 * - Show/hide loading overlay with status messages
 */

(function () {
    'use strict';

    // ---- DOM Elements ----
    const refInput = document.getElementById('refInput');
    const sketchInput = document.getElementById('sketchInput');
    const refDropZone = document.getElementById('refDropZone');
    const sketchDropZone = document.getElementById('sketchDropZone');
    const refPreview = document.getElementById('refPreview');
    const sketchPreview = document.getElementById('sketchPreview');
    const refPrompt = document.getElementById('refPrompt');
    const sketchPrompt = document.getElementById('sketchPrompt');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loaderStatus = document.getElementById('loaderStatus');
    const loaderBarFill = document.getElementById('loaderBarFill');
    const errorBanner = document.getElementById('errorBanner');
    const errorText = document.getElementById('errorText');
    const errorClose = document.getElementById('errorClose');
    const resultsSection = document.getElementById('resultsSection');
    const heatmapBtn = document.getElementById('heatmapBtn');
    const heatmapWrap = document.getElementById('heatmapWrap');
    const resetBtn = document.getElementById('resetBtn');

    // Track selected files
    let refFile = null;
    let sketchFile = null;
    let radarChartInstance = null;

    // ================================================================
    // FILE UPLOAD HANDLING
    // ================================================================

    function setupDropZone(dropZone, fileInput, previewImg, promptEl, onFileSet) {
        // Click to open file dialog
        dropZone.addEventListener('click', () => fileInput.click());

        // File selected via dialog
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0], previewImg, promptEl, dropZone, onFileSet);
            }
        });

        // Drag events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('active');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('active');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('active');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0], previewImg, promptEl, dropZone, onFileSet);
            }
        });
    }

    function handleFile(file, previewImg, promptEl, dropZone, onFileSet) {
        // Validate it's an image
        if (!file.type.startsWith('image/')) {
            showError('Please upload a valid image file (PNG, JPG, BMP, WebP).');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            promptEl.style.display = 'none';
            dropZone.classList.add('has-file');
        };
        reader.readAsDataURL(file);

        onFileSet(file);
        updateAnalyzeButton();
    }

    // Initialize drop zones
    setupDropZone(refDropZone, refInput, refPreview, refPrompt, (f) => { refFile = f; });
    setupDropZone(sketchDropZone, sketchInput, sketchPreview, sketchPrompt, (f) => { sketchFile = f; });

    function updateAnalyzeButton() {
        analyzeBtn.disabled = !(refFile && sketchFile);
    }

    // ================================================================
    // ANALYZE BUTTON
    // ================================================================

    analyzeBtn.addEventListener('click', async () => {
        if (!refFile || !sketchFile) return;
        await runAnalysis();
    });

    async function runAnalysis() {
        // Hide previous results/errors
        resultsSection.style.display = 'none';
        errorBanner.style.display = 'none';

        // Show loading overlay
        showLoading();

        // Build form data
        const formData = new FormData();
        formData.append('reference', refFile);
        formData.append('sketch', sketchFile);

        try {
            updateLoaderStatus('Uploading images...', 10);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            updateLoaderStatus('Processing analysis...', 60);

            const data = await response.json();

            if (!response.ok || data.error) {
                hideLoading();
                showError(data.error || 'An unexpected error occurred.');
                return;
            }

            updateLoaderStatus('Rendering results...', 90);

            // Short delay for UX feel
            await new Promise(r => setTimeout(r, 500));

            hideLoading();
            renderResults(data);

        } catch (err) {
            hideLoading();
            showError('Connection failed. Please ensure the server is running.');
            console.error(err);
        }
    }

    // ================================================================
    // LOADING OVERLAY
    // ================================================================

    const statusMessages = [
        'Detecting facial landmarks...',
        'Aligning faces...',
        'Analyzing edge structure...',
        'Computing proportions...',
        'Evaluating tonal accuracy...',
        'Generating heatmap...',
        'Preparing results...'
    ];

    let loadingInterval = null;

    function showLoading() {
        loadingOverlay.style.display = 'flex';
        loaderBarFill.style.width = '0%';
        let msgIndex = 0;
        loaderStatus.textContent = statusMessages[0];

        loadingInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % statusMessages.length;
            loaderStatus.textContent = statusMessages[msgIndex];
        }, 2000);
    }

    function updateLoaderStatus(msg, pct) {
        loaderStatus.textContent = msg;
        loaderBarFill.style.width = pct + '%';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }

    // ================================================================
    // ERROR HANDLING
    // ================================================================

    function showError(message) {
        errorText.textContent = message;
        errorBanner.style.display = 'flex';
        // Scroll to error
        errorBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    errorClose.addEventListener('click', () => {
        errorBanner.style.display = 'none';
    });

    // ================================================================
    // RENDER RESULTS
    // ================================================================

    function renderResults(data) {
        resultsSection.style.display = 'block';

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        // Animate scores
        animateCircularProgress('overallRing', 'overallValue', data.final_score, 52, true);
        animateCircularProgress('structuralRing', 'structuralValue', data.structural_score, 34);
        animateCircularProgress('proportionRing', 'proportionValue', data.proportion_score, 34);
        animateCircularProgress('tonalRing', 'tonalValue', data.tonal_score, 34);

        // Grade badge
        const gradeBadge = document.getElementById('gradeBadge');
        gradeBadge.textContent = data.grade;

        // Radar chart
        renderRadarChart(data);

        // Image comparison
        document.getElementById('compRefImg').src = data.ref_image_url;
        document.getElementById('compSketchImg').src = data.sketch_image_url;

        // Heatmap
        const heatmapImg = document.getElementById('heatmapImg');
        heatmapImg.src = data.heatmap_url;
        heatmapWrap.style.display = 'none';
        heatmapBtn.classList.remove('active');
        heatmapBtn.innerHTML = '<span>🔥</span> Show Error Heatmap';

        // Proportion details table
        renderProportionTable(data.proportion_details);

        // Tonal region details
        renderTonalRegions(data.tonal_details);

        // Feedback
        renderFeedback(data.feedback);
    }

    // ================================================================
    // CIRCULAR PROGRESS ANIMATION
    // ================================================================

    function animateCircularProgress(ringId, valueId, targetPct, radius, isLarge) {
        const ring = document.getElementById(ringId);
        const valueEl = document.getElementById(valueId);
        const circumference = 2 * Math.PI * radius;

        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = circumference;

        // Add gradient def for large ring if needed
        if (isLarge && !document.getElementById('mainGrad')) {
            const svg = ring.closest('svg');
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            defs.innerHTML = `
                <linearGradient id="mainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#00e5ff"/>
                    <stop offset="50%" stop-color="#b388ff"/>
                    <stop offset="100%" stop-color="#ff4081"/>
                </linearGradient>
            `;
            svg.prepend(defs);
            ring.style.stroke = 'url(#mainGrad)';
        }

        // Animate after short delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                const offset = circumference - (targetPct / 100) * circumference;
                ring.style.strokeDashoffset = offset;
            }, 100);
        });

        // Animate counter
        animateCounter(valueEl, 0, targetPct, 1200);
    }

    function animateCounter(el, start, end, duration) {
        const startTime = performance.now();
        const diff = end - start;

        function tick(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + diff * eased;
            el.textContent = Math.round(current) + '%';

            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        }
        requestAnimationFrame(tick);
    }

    // ================================================================
    // RADAR CHART
    // ================================================================

    function renderRadarChart(data) {
        const canvas = document.getElementById('radarChart');

        // Destroy previous chart instance
        if (radarChartInstance) {
            radarChartInstance.destroy();
        }

        radarChartInstance = new Chart(canvas, {
            type: 'radar',
            data: {
                labels: ['Structural', 'Proportion', 'Tonal'],
                datasets: [{
                    label: 'Score',
                    data: [data.structural_score, data.proportion_score, data.tonal_score],
                    backgroundColor: 'rgba(0, 229, 255, 0.1)',
                    borderColor: 'rgba(0, 229, 255, 0.7)',
                    pointBackgroundColor: ['#00e5ff', '#b388ff', '#ff4081'],
                    pointBorderColor: '#0a0a0f',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            color: 'rgba(255,255,255,0.3)',
                            backdropColor: 'transparent',
                            font: { size: 10 }
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.06)',
                        },
                        angleLines: {
                            color: 'rgba(255,255,255,0.06)',
                        },
                        pointLabels: {
                            color: 'rgba(255,255,255,0.7)',
                            font: { size: 13, weight: '500' }
                        }
                    }
                },
                plugins: {
                    legend: { display: false }
                },
                animation: {
                    duration: 1200,
                    easing: 'easeOutCubic'
                }
            }
        });
    }

    // ================================================================
    // PROPORTION TABLE
    // ================================================================

    function renderProportionTable(details) {
        const tbody = document.getElementById('proportionTableBody');
        tbody.innerHTML = '';

        if (!details || details.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">No proportion data available.</td></tr>';
            return;
        }

        for (const d of details) {
            const tr = document.createElement('tr');
            const accClass = d.accuracy >= 80 ? 'accuracy-high' :
                d.accuracy >= 60 ? 'accuracy-mid' : 'accuracy-low';

            tr.innerHTML = `
                <td>${d.name}</td>
                <td>${d.reference.toFixed(3)}</td>
                <td>${d.sketch.toFixed(3)}</td>
                <td>${d.deviation_pct.toFixed(1)}%</td>
                <td class="${accClass}">${d.accuracy.toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        }
    }

    // ================================================================
    // TONAL REGIONS
    // ================================================================

    function renderTonalRegions(details) {
        const container = document.getElementById('tonalRegions');
        container.innerHTML = '';

        if (!details || details.length === 0) {
            container.innerHTML = '<p style="color:var(--text-muted);">No tonal data available.</p>';
            return;
        }

        for (const d of details) {
            const card = document.createElement('div');
            card.className = 'tonal-region-card';

            const barColor = d.accuracy >= 80 ? 'var(--green)' :
                d.accuracy >= 60 ? 'var(--yellow)' : 'var(--red)';

            card.innerHTML = `
                <div class="tonal-region-name">${d.region}</div>
                <div class="tonal-region-bar">
                    <div class="tonal-region-bar-fill" style="width:0%;background:${barColor};"></div>
                </div>
                <div class="tonal-region-feedback">${d.feedback || ''}</div>
            `;
            container.appendChild(card);

            // Animate bar
            requestAnimationFrame(() => {
                setTimeout(() => {
                    card.querySelector('.tonal-region-bar-fill').style.width = d.accuracy + '%';
                }, 200);
            });
        }
    }

    // ================================================================
    // FEEDBACK
    // ================================================================

    function renderFeedback(feedbackList) {
        const container = document.getElementById('feedbackList');
        container.innerHTML = '';

        if (!feedbackList || feedbackList.length === 0) {
            container.innerHTML = '<p style="color:var(--text-muted);">No feedback available.</p>';
            return;
        }

        for (const text of feedbackList) {
            const item = document.createElement('div');
            item.className = 'feedback-item';
            item.textContent = text;
            container.appendChild(item);
        }
    }

    // ================================================================
    // HEATMAP TOGGLE
    // ================================================================

    heatmapBtn.addEventListener('click', () => {
        const isVisible = heatmapWrap.style.display !== 'none';
        heatmapWrap.style.display = isVisible ? 'none' : 'block';
        heatmapBtn.classList.toggle('active', !isVisible);
        heatmapBtn.innerHTML = isVisible
            ? '<span>🔥</span> Show Error Heatmap'
            : '<span>🔥</span> Hide Error Heatmap';
    });

    // ================================================================
    // RESET BUTTON
    // ================================================================

    resetBtn.addEventListener('click', () => {
        // Hide results
        resultsSection.style.display = 'none';

        // Reset file selections
        refFile = null;
        sketchFile = null;
        refInput.value = '';
        sketchInput.value = '';
        refPreview.style.display = 'none';
        sketchPreview.style.display = 'none';
        refPrompt.style.display = 'flex';
        sketchPrompt.style.display = 'flex';
        refDropZone.classList.remove('has-file');
        sketchDropZone.classList.remove('has-file');

        updateAnalyzeButton();

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

})();
