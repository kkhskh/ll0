// HEPHAELION System - Core JavaScript Functionality

// Global variables for system state
const systemState = {
    activeSatellites: 24,
    totalSatellites: 25,
    systemLoad: 42,
    securityStatus: 'Active',
    lastUpdate: new Date(),
    activeDisasters: new Map(),
    simulationMode: 'current'
};

// ML Model Configuration
const mlConfig = {
    modelVersions: {
        hurricane: 'v2.1',
        tsunami: 'v1.9',
        earthquake: 'v2.0',
        wildfire: 'v2.3',
        flood: 'v1.8',
        volcanic: 'v1.5',
        hailstorm: 'v1.7'
    },
    accuracy: {
        overall: 0.95,
        hurricane: 0.96,
        tsunami: 0.94,
        earthquake: 0.93,
        wildfire: 0.97,
        flood: 0.95,
        volcanic: 0.92,
        hailstorm: 0.94
    },
    transferLearningEnabled: true,
    crossValidationFolds: 5,
    batchSize: 32,
    epochsPerDisaster: 100
};

// Satellite Network Configuration
const satelliteConfig = {
    totalSatellites: 25,
    orbitHeight: 'LEO',
    coverage: {
        global: true,
        updateFrequency: '1min'
    },
    sensors: {
        infrared: true,
        optical: true,
        sar: true,
        multispectral: true
    },
    dataTransmission: {
        bandwidth: '1.2GB/s',
        latency: '100ms'
    }
};

// Initialize canvas contexts
let globalMapCtx;
let pressureChartCtx;
let tempChartCtx;

// ESRI Map Configuration
const mapConfig = {
    basemap: 'satellite',
    center: [0, 0],
    zoom: 2,
    container: 'globalMap',
    spatialReference: {
        wkid: 4326
    }
};

// Add after existing canvas contexts
let esriMap;
let mapView;
let graphicsLayer;
const apiBase = window.HEPHAELION_API_BASE || 'http://127.0.0.1:8000';
const investorState = {
    selectedMonth: new Date().getUTCMonth() + 1,
    summary: null,
    monthly: null,
    watchlist: null
};

// Disaster types and their characteristics
const disasterTypes = {
    hurricane: {
        color: 'rgba(100, 200, 255, 0.7)',
        icon: 'wind',
        severity: 'high'
    },
    tsunami: {
        color: 'rgba(100, 150, 255, 0.7)',
        icon: 'water',
        severity: 'critical'
    },
    earthquake: {
        color: 'rgba(255, 150, 100, 0.7)',
        icon: 'mountain',
        severity: 'high'
    },
    wildfire: {
        color: 'rgba(255, 100, 50, 0.7)',
        icon: 'fire',
        severity: 'high'
    },
    storm: {
        color: 'rgba(200, 200, 255, 0.7)',
        icon: 'bolt',
        severity: 'medium'
    }
};

// Satellite Data Processing Pipeline
const dataPipeline = {
    preprocessors: new Map([
        ['sar', data => processSARData(data)],
        ['infrared', data => processInfraredData(data)],
        ['optical', data => processOpticalData(data)],
        ['multispectral', data => processMultispectralData(data)]
    ]),
    analyzers: new Map([
        ['pattern_recognition', data => analyzePatterns(data)],
        ['anomaly_detection', data => detectAnomalies(data)],
        ['risk_assessment', data => assessRisk(data)]
    ]),
    predictors: new Map([
        ['short_term', data => predictShortTerm(data)],
        ['medium_term', data => predictMediumTerm(data)],
        ['long_term', data => predictLongTerm(data)]
    ])
};

// Initialize the system when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    initializeCanvases();
    initializeEventListeners();
    if (isInvestorMode()) {
        await initializeInvestorDashboard();
        return;
    }
    startSystemMonitoring();
    updateSystemStatus();
    
    // Initialize ML models
    initializeMLModels();
    
    // Initialize satellite network
    initializeSatelliteNetwork();
});

// Canvas initialization
function initializeCanvases() {
    // Initialize ESRI map
    initializeESRIMap();

    // Initialize charts
    const pressureChart = document.getElementById('pressureChart');
    if (pressureChart) {
        pressureChartCtx = pressureChart.getContext('2d');
        resizeCanvas(pressureChart);
    }

    const tempChart = document.getElementById('tempChart');
    if (tempChart) {
        tempChartCtx = tempChart.getContext('2d');
        resizeCanvas(tempChart);
    }
}

// Canvas resize handling
function resizeCanvas(canvas) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
}

// Initialize event listeners
function initializeEventListeners() {
    // Simulation control buttons
    if (!isInvestorMode()) {
        document.querySelectorAll('.sim-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.id.replace('Btn', '');
                activateSimulationMode(mode);
            });
        });
    }

    // Alert handling
    document.getElementById('dismissAlert')?.addEventListener('click', dismissAlert);
    document.getElementById('closeAlert')?.addEventListener('click', dismissAlert);

    // Response option buttons
    document.querySelectorAll('.response-btn').forEach(btn => {
        btn.addEventListener('click', handleResponseAction);
    });

    // Window resize handling
    window.addEventListener('resize', handleResize);

    document.getElementById('refreshInvestorData')?.addEventListener('click', async () => {
        await initializeInvestorDashboard(true);
    });
    document.getElementById('openInvestorDocs')?.addEventListener('click', () => {
        window.open(`${apiBase}/docs`, '_blank');
    });
    document.getElementById('openRunSummary')?.addEventListener('click', () => {
        window.open(`${apiBase}/api/v1/wildfire/investor/summary`, '_blank');
    });
}

// Simulation mode activation
function activateSimulationMode(mode) {
    systemState.simulationMode = mode;
    resetSimulationButtons();
    document.getElementById(`${mode}Btn`)?.classList.add('active');

    switch (mode) {
        case 'current':
            drawCurrentState();
            break;
        case 'hurricane':
            simulateHurricane();
            break;
        case 'tsunami':
            simulateTsunami();
            break;
        case 'storm':
            simulateStorms();
            break;
        case 'wildfire':
            simulateWildfire();
            break;
    }
}

// Reset simulation button states
function resetSimulationButtons() {
    document.querySelectorAll('.sim-btn').forEach(btn => {
        btn.classList.remove('active');
    });
}

// Draw current state on global map
function drawCurrentState() {
    if (!graphicsLayer) return;
    
    graphicsLayer.removeAll();
    initializeSatelliteMarkers();
    drawActiveDisasters();
}

// World map drawing
function drawWorldMap() {
    if (!globalMapCtx) return;

    const ctx = globalMapCtx;
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;

    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Simplified world map outline
    // North America
    ctx.moveTo(width * 0.1, height * 0.3);
    ctx.lineTo(width * 0.25, height * 0.2);
    ctx.lineTo(width * 0.25, height * 0.5);
    ctx.lineTo(width * 0.15, height * 0.6);
    ctx.lineTo(width * 0.1, height * 0.3);

    // South America
    ctx.moveTo(width * 0.25, height * 0.5);
    ctx.lineTo(width * 0.3, height * 0.8);
    ctx.lineTo(width * 0.22, height * 0.85);
    ctx.lineTo(width * 0.15, height * 0.6);

    // Europe & Africa
    ctx.moveTo(width * 0.4, height * 0.2);
    ctx.lineTo(width * 0.45, height * 0.4);
    ctx.lineTo(width * 0.4, height * 0.8);
    ctx.lineTo(width * 0.5, height * 0.7);
    ctx.lineTo(width * 0.55, height * 0.4);
    ctx.lineTo(width * 0.5, height * 0.25);
    ctx.lineTo(width * 0.4, height * 0.2);

    // Asia & Australia
    ctx.moveTo(width * 0.5, height * 0.25);
    ctx.lineTo(width * 0.8, height * 0.25);
    ctx.lineTo(width * 0.85, height * 0.5);
    ctx.lineTo(width * 0.75, height * 0.6);
    ctx.lineTo(width * 0.55, height * 0.4);

    ctx.moveTo(width * 0.75, height * 0.6);
    ctx.lineTo(width * 0.85, height * 0.8);
    ctx.lineTo(width * 0.75, height * 0.85);
    ctx.lineTo(width * 0.65, height * 0.7);
    ctx.lineTo(width * 0.75, height * 0.6);

    ctx.stroke();

    // Grid lines
    drawGrid(ctx);
}

// Draw grid lines
function drawGrid(ctx) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;

    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.5;

    // Latitude lines
    for (let i = 1; i < 10; i++) {
        ctx.beginPath();
        ctx.moveTo(0, height * i/10);
        ctx.lineTo(width, height * i/10);
        ctx.stroke();
    }

    // Longitude lines
    for (let i = 1; i < 10; i++) {
        ctx.beginPath();
        ctx.moveTo(width * i/10, 0);
        ctx.lineTo(width * i/10, height);
        ctx.stroke();
    }
}

// Simulate hurricane
function simulateHurricane() {
    if (!globalMapCtx) return;

    drawCurrentState();
    
    const ctx = globalMapCtx;
    const centerX = ctx.canvas.width * 0.2;
    const centerY = ctx.canvas.height * 0.4;
    const radius = 30;

    // Draw hurricane spiral
    for (let i = 0; i < 5; i++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius - i * 5, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(100, 200, 255, ${0.7 - i * 0.1})`;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Add to active disasters
    systemState.activeDisasters.set('hurricane', {
        type: 'hurricane',
        location: { x: centerX, y: centerY },
        severity: 'high',
        timestamp: new Date()
    });

    updateSystemStatus();
}

// Simulate tsunami
function simulateTsunami() {
    if (!globalMapCtx) return;

    drawCurrentState();

    const ctx = globalMapCtx;
    const centerX = ctx.canvas.width * 0.7;
    const centerY = ctx.canvas.height * 0.6;

    // Draw tsunami wave fronts
    for (let i = 1; i < 6; i++) {
        const radius = i * 30;
        const opacity = 1 - i * 0.15;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(100, 150, 255, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Add to active disasters
    systemState.activeDisasters.set('tsunami', {
        type: 'tsunami',
        location: { x: centerX, y: centerY },
        severity: 'critical',
        timestamp: new Date()
    });

    updateSystemStatus();
    showTsunamiAlert();
}

// Simulate storms
function simulateStorms() {
    if (!globalMapCtx) return;

    drawCurrentState();

    const stormLocations = [
        { x: 0.3, y: 0.4 },
        { x: 0.6, y: 0.3 },
        { x: 0.5, y: 0.7 }
    ];

    stormLocations.forEach(loc => {
        drawStormCell(
            globalMapCtx.canvas.width * loc.x,
            globalMapCtx.canvas.height * loc.y,
            25
        );
    });

    // Add to active disasters
    systemState.activeDisasters.set('storm', {
        type: 'storm',
        locations: stormLocations,
        severity: 'medium',
        timestamp: new Date()
    });

    updateSystemStatus();
}

// Draw storm cell
function drawStormCell(x, y, size) {
    if (!globalMapCtx) return;

    const ctx = globalMapCtx;

    // Storm cloud
    ctx.fillStyle = 'rgba(200, 200, 255, 0.2)';
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();

    // Lightning bolt
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y - size/2);
    ctx.lineTo(x + size/4, y);
    ctx.lineTo(x - size/3, y + size/3);
    ctx.lineTo(x, y + size);
    ctx.stroke();
}

// Simulate wildfire
function simulateWildfire() {
    if (!globalMapCtx) return;

    drawCurrentState();

    const ctx = globalMapCtx;
    const centerX = ctx.canvas.width * 0.4;
    const centerY = ctx.canvas.height * 0.5;

    // Draw fire spread
    const gradient = ctx.createRadialGradient(
        centerX, centerY, 0,
        centerX, centerY, 50
    );
    gradient.addColorStop(0, 'rgba(255, 100, 50, 0.8)');
    gradient.addColorStop(0.5, 'rgba(255, 150, 50, 0.4)');
    gradient.addColorStop(1, 'rgba(255, 200, 50, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 50, 0, Math.PI * 2);
    ctx.fill();

    // Add to active disasters
    systemState.activeDisasters.set('wildfire', {
        type: 'wildfire',
        location: { x: centerX, y: centerY },
        severity: 'high',
        timestamp: new Date()
    });

    updateSystemStatus();
}

// Draw satellite network
function drawSatelliteNetwork() {
    const satellites = document.querySelectorAll('.satellite-indicator');
    satellites.forEach(sat => {
        const isOffline = sat.classList.contains('offline');
        if (!isOffline) {
            const rect = sat.getBoundingClientRect();
            const x = rect.left + rect.width / 2;
            const y = rect.top + rect.height / 2;
            
            // Draw satellite coverage area
            if (globalMapCtx) {
                globalMapCtx.beginPath();
                globalMapCtx.arc(x, y, 50, 0, Math.PI * 2);
                globalMapCtx.fillStyle = 'rgba(68, 136, 255, 0.1)';
                globalMapCtx.fill();
            }
        }
    });
}

// Update system status
function updateSystemStatus() {
    // Update header status
    document.querySelector('.satellite-status .status-value').textContent = 
        `${systemState.activeSatellites}/${systemState.totalSatellites}`;

    // Update footer status
    document.querySelector('.footer-status .status-item:first-child').textContent = 
        `Satellites: ${systemState.activeSatellites}/${systemState.totalSatellites}`;

    // Update system load
    document.querySelector('.footer-status .status-item:nth-child(3)').textContent = 
        `System Load: ${systemState.systemLoad}%`;

    // Update charts
    updateCharts();

    // Add ML model status updates
    updateMLModelStatus();
    
    // Add satellite network status
    updateSatelliteNetworkStatus();
    
    // Process real-time data
    processRealTimeData();
}

// Update atmospheric and temperature charts
function updateCharts() {
    if (pressureChartCtx) drawChart(pressureChartCtx, 'pressure');
    if (tempChartCtx) drawChart(tempChartCtx, 'temperature');
}

// Draw chart
function drawChart(ctx, type = 'pressure') {
    clearCanvas(ctx);
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;

    // Draw grid
    drawChartGrid(ctx);

    // Generate data points
    const data = generateChartData(type);

    // Plot line
    ctx.strokeStyle = type === 'pressure' ? '#4488ff' : '#ff4444';
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((value, index) => {
        const x = (width / data.length) * index;
        const y = height - (height * (value / 100));

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Draw threshold line for temperature
    if (type === 'temperature') {
        drawTemperatureThreshold(ctx);
    }
}

// Generate chart data
function generateChartData(type) {
    const data = [];
    for (let i = 0; i < 20; i++) {
        if (type === 'pressure') {
            // Cyclical pattern with anomaly
            data.push(Math.sin(i/3) * 20 + 50 + (i > 15 ? 15 : 0));
        } else {
            // Rising temperature with fluctuations
            data.push(40 + i * 1.5 + Math.random() * 5);
        }
    }
    return data;
}

// Draw chart grid
function drawChartGrid(ctx) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;

    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.5;

    // Horizontal grid lines
    for (let i = 1; i < 5; i++) {
        ctx.beginPath();
        ctx.moveTo(0, height * i/5);
        ctx.lineTo(width, height * i/5);
        ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 1; i < 10; i++) {
        ctx.beginPath();
        ctx.moveTo(width * i/10, 0);
        ctx.lineTo(width * i/10, height);
        ctx.stroke();
    }
}

// Draw temperature threshold line
function drawTemperatureThreshold(ctx) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;

    ctx.strokeStyle = 'rgba(255, 100, 100, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height * 0.2);
    ctx.lineTo(width, height * 0.2);
    ctx.stroke();

    // Label
    ctx.fillStyle = 'rgba(255, 100, 100, 0.7)';
    ctx.font = '10px Courier New';
    ctx.fillText('CRITICAL THRESHOLD', 10, height * 0.18);
}

// Clear canvas
function clearCanvas(ctx) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

// Handle window resize
function handleResize() {
    initializeCanvases();
    if (isInvestorMode()) {
        renderInvestorCharts();
        renderInvestorMap();
        return;
    }
    drawCurrentState();
    updateCharts();
}

// Show tsunami alert
function showTsunamiAlert() {
    const alert = document.getElementById('tsunamiAlert');
    if (alert) {
        alert.style.display = 'block';
        alert.style.animation = 'slideIn 0.3s ease-out';
    }
}

// Dismiss alert
function dismissAlert() {
    const alert = document.getElementById('tsunamiAlert');
    if (alert) {
        alert.style.display = 'none';
    }
}

// Handle response action
function handleResponseAction(e) {
    const btn = e.currentTarget;
    const level = btn.classList.contains('level-4') ? 'CRITICAL' :
                 btn.classList.contains('level-3') ? 'HIGH' :
                 btn.classList.contains('level-2') ? 'MEDIUM' : 'LOW';

    console.log(`Response level ${level} activated`);
    // Implement response action logic here
}

// Start system monitoring
function startSystemMonitoring() {
    // Update system status every 5 seconds
    setInterval(() => {
        systemState.systemLoad = Math.floor(Math.random() * 30) + 30; // 30-60% load
        updateSystemStatus();
    }, 5000);

    // Simulate random events
    setInterval(() => {
        if (Math.random() < 0.1) { // 10% chance of event
            const events = ['hurricane', 'tsunami', 'storm', 'wildfire'];
            const randomEvent = events[Math.floor(Math.random() * events.length)];
            activateSimulationMode(randomEvent);
        }
    }, 30000);

    // Add ML model monitoring
    setInterval(() => {
        updateMLModelStatus();
        runModelValidation();
    }, 60000); // Every minute

    // Add satellite data processing
    setInterval(() => {
        processRealTimeData();
    }, 1000); // Every second

    // Add predictive analysis
    setInterval(() => {
        runPredictiveAnalysis();
    }, 300000); // Every 5 minutes
}

// Add new monitoring functions
function runModelValidation() {
    Object.keys(mlConfig.modelVersions).forEach(disasterType => {
        validateModel(disasterType);
    });
}

function runPredictiveAnalysis() {
    const predictions = {
        shortTerm: predictShortTerm(collectSatelliteData()),
        mediumTerm: predictMediumTerm(collectHistoricalData()),
        longTerm: predictLongTerm(aggregateData())
    };

    updatePredictionDisplays(predictions);
}

// Add after existing handleResponseAction function
function handleDisasterPrediction(prediction) {
    if (prediction.confidence > 0.9) {
        // High confidence prediction
        triggerEarlyWarning(prediction);
        notifyAuthorities(prediction);
        initializeResponseProtocol(prediction);
    } else if (prediction.confidence > 0.7) {
        // Medium confidence prediction
        increaseSurveillance(prediction.region);
        alertMonitoringTeam(prediction);
    }
}

// Add new status update functions
function updateMLModelStatus() {
    const modelStatus = document.querySelector('.ml-model-status');
    if (modelStatus) {
        modelStatus.textContent = `ML Models: ${calculateModelAccuracy()}% Accuracy`;
    }
}

function updateSatelliteNetworkStatus() {
    const networkStatus = document.querySelector('.satellite-network-status');
    if (networkStatus) {
        networkStatus.textContent = `Network: ${systemState.activeSatellites}/${satelliteConfig.totalSatellites} Satellites Active`;
    }
}

function processRealTimeData() {
    // Process data from all active satellites
    const satelliteData = collectSatelliteData();
    
    // Run data through processing pipeline
    const processedData = processSatelliteData(satelliteData);
    
    // Update visualizations
    updateVisualizations(processedData);
    
    // Check for potential disasters
    checkForDisasters(processedData);
}

function initializeMLModels() {
    Object.keys(mlConfig.modelVersions).forEach(disasterType => {
        loadModel(disasterType);
    });
}

function initializeSatelliteNetwork() {
    configureSatellites();
    establishDataLinks();
    startDataCollection();
}

// Add ESRI map initialization
function initializeESRIMap() {
    require([
        "esri/Map",
        "esri/views/MapView",
        "esri/layers/GraphicsLayer",
        "esri/Graphic",
        "esri/geometry/Point",
        "esri/symbols/SimpleMarkerSymbol",
        "esri/symbols/SimpleLineSymbol",
        "esri/symbols/SimpleFillSymbol"
    ], function(Map, MapView, GraphicsLayer, Graphic, Point, SimpleMarkerSymbol, SimpleLineSymbol, SimpleFillSymbol) {
        // Create the map
        esriMap = new Map({
            basemap: mapConfig.basemap
        });

        // Create the view
        mapView = new MapView({
            container: mapConfig.container,
            map: esriMap,
            center: mapConfig.center,
            zoom: mapConfig.zoom,
            spatialReference: mapConfig.spatialReference
        });

        // Create a graphics layer for disasters
        graphicsLayer = new GraphicsLayer();
        esriMap.add(graphicsLayer);

        // Store constructors for later use
        window.esriConstructors = {
            Graphic,
            Point,
            SimpleMarkerSymbol,
            SimpleLineSymbol,
            SimpleFillSymbol
        };

        // Initialize satellite markers
        initializeSatelliteMarkers();
    });
}

// Add satellite markers to the map
function initializeSatelliteMarkers() {
    if (!window.esriConstructors) return;

    const { Graphic, Point, SimpleMarkerSymbol } = window.esriConstructors;

    // Define satellite positions (example coordinates)
    const satellitePositions = [
        { longitude: -100, latitude: 40 },
        { longitude: -20, latitude: 30 },
        { longitude: 60, latitude: 45 },
        { longitude: 140, latitude: 35 },
        { longitude: -60, latitude: -20 }
    ];

    satellitePositions.forEach((pos, index) => {
        const point = new Point({
            longitude: pos.longitude,
            latitude: pos.latitude
        });

        const symbol = new SimpleMarkerSymbol({
            style: "circle",
            color: systemState.activeSatellites > index ? [68, 136, 255, 0.8] : [136, 136, 136, 0.8],
            size: "12px",
            outline: {
                color: [255, 255, 255, 0.5],
                width: 1
            }
        });

        const graphic = new Graphic({
            geometry: point,
            symbol: symbol,
            attributes: {
                type: "satellite",
                id: `SAT-${index + 1}`,
                status: systemState.activeSatellites > index ? "active" : "inactive"
            }
        });

        graphicsLayer.add(graphic);
    });
}

// Modify drawActiveDisasters function
function drawActiveDisasters() {
    if (!window.esriConstructors || !graphicsLayer) return;

    const { Graphic, Point, SimpleMarkerSymbol, SimpleFillSymbol } = window.esriConstructors;

    systemState.activeDisasters.forEach((disaster, type) => {
        const disasterConfig = disasterTypes[type];
        if (!disasterConfig) return;

        switch (type) {
            case 'hurricane':
                drawHurricaneGraphic(disaster.location);
                break;
            case 'tsunami':
                drawTsunamiGraphic(disaster.location);
                break;
            case 'storm':
                disaster.locations.forEach(loc => drawStormGraphic(loc));
                break;
            case 'wildfire':
                drawWildfireGraphic(disaster.location);
                break;
        }
    });
}

// Add disaster visualization functions
function drawHurricaneGraphic(location) {
    if (!window.esriConstructors) return;

    const { Graphic, Point, SimpleMarkerSymbol } = window.esriConstructors;

    const point = new Point({
        longitude: location.x,
        latitude: location.y
    });

    const symbol = new SimpleMarkerSymbol({
        style: "circle",
        color: [100, 200, 255, 0.5],
        size: "30px",
        outline: {
            color: [100, 200, 255, 0.8],
            width: 2
        }
    });

    const graphic = new Graphic({
        geometry: point,
        symbol: symbol,
        attributes: {
            type: "hurricane"
        }
    });

    graphicsLayer.add(graphic);
}

function drawTsunamiGraphic(location) {
    if (!window.esriConstructors) return;

    const { Graphic, Point, SimpleFillSymbol } = window.esriConstructors;

    const point = new Point({
        longitude: location.x,
        latitude: location.y
    });

    const symbol = new SimpleFillSymbol({
        color: [100, 150, 255, 0.3],
        outline: {
            color: [100, 150, 255, 0.8],
            width: 2
        }
    });

    const graphic = new Graphic({
        geometry: point,
        symbol: symbol,
        attributes: {
            type: "tsunami"
        }
    });

    graphicsLayer.add(graphic);
}

function drawStormGraphic(location) {
    if (!window.esriConstructors) return;

    const { Graphic, Point, SimpleMarkerSymbol } = window.esriConstructors;

    const point = new Point({
        longitude: location.x,
        latitude: location.y
    });

    const symbol = new SimpleMarkerSymbol({
        style: "square",
        color: [200, 200, 255, 0.5],
        size: "20px",
        outline: {
            color: [255, 255, 255, 0.8],
            width: 1
        }
    });

    const graphic = new Graphic({
        geometry: point,
        symbol: symbol,
        attributes: {
            type: "storm"
        }
    });

    graphicsLayer.add(graphic);
}

function drawWildfireGraphic(location) {
    if (!window.esriConstructors) return;

    const { Graphic, Point, SimpleMarkerSymbol } = window.esriConstructors;

    const point = new Point({
        longitude: location.x,
        latitude: location.y
    });

    const symbol = new SimpleMarkerSymbol({
        style: "diamond",
        color: [255, 100, 50, 0.7],
        size: "25px",
        outline: {
            color: [255, 150, 50, 0.8],
            width: 1
        }
    });

    const graphic = new Graphic({
        geometry: point,
        symbol: symbol,
        attributes: {
            type: "wildfire"
        }
    });

    graphicsLayer.add(graphic);
}

function isInvestorMode() {
    return document.body?.dataset.dashboardMode === 'wildfire-investor';
}

async function fetchJson(path) {
    const response = await fetch(`${apiBase}${path}`);
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
    }
    return await response.json();
}

function configureInvestorMonthButtons() {
    const monthButtons = [
        { id: 'currentState', label: 'Current', month: new Date().getUTCMonth() + 1 },
        { id: 'hurricaneBtn', label: 'Jan', month: 1 },
        { id: 'tsunamiBtn', label: 'Apr', month: 4 },
        { id: 'wildfireBtn', label: 'Jul', month: 7 },
        { id: 'stormBtn', label: 'Oct', month: 10 }
    ];

    monthButtons.forEach(({ id, label, month }) => {
        const button = document.getElementById(id);
        if (!button) return;
        button.innerHTML = `<i class="fas fa-fire"></i> ${label}`;
        button.classList.toggle('active', investorState.selectedMonth === month);
        button.onclick = async () => {
            investorState.selectedMonth = month;
            monthButtons.forEach(({ id: otherId, month: otherMonth }) => {
                document.getElementById(otherId)?.classList.toggle('active', otherMonth === month);
            });
            await loadInvestorMonth(month);
        };
    });
}

async function initializeInvestorDashboard(forceRefresh = false) {
    configureInvestorMonthButtons();
    document.querySelector('.map-overlay')?.setAttribute('style', 'display:none;');
    document.getElementById('tsunamiAlert')?.setAttribute('style', 'display:none;');

    if (forceRefresh || !investorState.summary) {
        investorState.summary = await fetchJson('/api/v1/wildfire/investor/summary');
    }

    renderInvestorSummary();
    renderProofMetrics();
    await loadInvestorMonth(investorState.selectedMonth);
}

async function loadInvestorMonth(month) {
    investorState.monthly = await fetchJson(`/api/v1/wildfire/risk/monthly?month=${month}&limit=20`);
    investorState.watchlist = await fetchJson(`/api/v1/wildfire/investor/watchlist?month=${month}&limit=20`);
    renderWatchlistSignals();
    renderInvestorCharts();
    renderInvestorMap();
}

function renderInvestorSummary() {
    const summary = investorState.summary;
    if (!summary) return;

    document.getElementById('activeRunValue').textContent = summary.run_name;
    document.getElementById('globalStatusValue').textContent = (
        summary.presentation_guidance?.recommended_mode || 'RANKED'
    ).replaceAll('_', ' ').toUpperCase();
    document.getElementById('guidanceBanner').textContent =
        summary.presentation_guidance?.warning || 'Use as ranked wildfire signal.';
    document.getElementById('runStatusBadge').textContent =
        summary.presentation_guidance?.recommended_score_field || 'relative_risk_score';

    const proof = summary.proof_points || {};
    const snapshot = document.querySelector('.satellite-grid');
    if (snapshot) {
        snapshot.innerHTML = `
            <div class="satellite-item online">
                <div class="satellite-icon"><i class="fas fa-fire"></i></div>
                <div class="satellite-info">
                    <div class="satellite-name">ROC AUC</div>
                    <div class="satellite-status">${formatMetric(proof.roc_auc)}</div>
                </div>
            </div>
            <div class="satellite-item online">
                <div class="satellite-icon"><i class="fas fa-chart-line"></i></div>
                <div class="satellite-info">
                    <div class="satellite-name">PR AUC</div>
                    <div class="satellite-status">${formatMetric(proof.pr_auc)}</div>
                </div>
            </div>
            <div class="satellite-item warning">
                <div class="satellite-icon"><i class="fas fa-filter"></i></div>
                <div class="satellite-info">
                    <div class="satellite-name">Top 10% Precision</div>
                    <div class="satellite-status">${formatMetric(proof.precision_top_10pct)}</div>
                </div>
            </div>
            <div class="satellite-item online">
                <div class="satellite-icon"><i class="fas fa-bullseye"></i></div>
                <div class="satellite-info">
                    <div class="satellite-name">Top 10% Capture</div>
                    <div class="satellite-status">${formatMetric(proof.capture_top_10pct)}</div>
                </div>
            </div>
        `;
    }
}

function renderProofMetrics() {
    const summary = investorState.summary;
    const tbody = document.getElementById('proofMetricsBody');
    if (!summary || !tbody) return;

    const proof = summary.proof_points || {};
    const guidance = summary.presentation_guidance || {};
    const rows = [
        ['ROC AUC', formatMetric(proof.roc_auc), 'Ranking quality across all cells.', 'Good for relative discrimination.'],
        ['PR AUC', formatMetric(proof.pr_auc), 'More informative under class imbalance.', 'Useful for hotspot screening.'],
        ['Top 10% Precision', formatMetric(proof.precision_top_10pct), 'How many positives appear in the top-decile signal bucket.', 'Best KPI for watchlist triage.'],
        ['Brier Score', formatMetric(proof.brier_score), 'Calibration and probability quality.', guidance.warning || 'Do not treat raw probabilities literally yet.']
    ];

    tbody.innerHTML = rows.map(([metric, value, why, note]) => `
        <tr>
            <td>${metric}</td>
            <td>${value}</td>
            <td>${why}</td>
            <td>${note}</td>
        </tr>
    `).join('');
}

function renderWatchlistSignals() {
    const tbody = document.getElementById('watchlistSignalsBody');
    if (!tbody) return;
    const matches = investorState.watchlist?.watchlist_matches || [];
    if (!matches.length) {
        tbody.innerHTML = '<tr class="alert-row low"><td colspan="6">No watchlist names matched the selected month.</td></tr>';
        return;
    }
    tbody.innerHTML = matches.map((item) => `
        <tr class="alert-row medium">
            <td>${item.symbol}</td>
            <td>${item.region}</td>
            <td>${item.theme}</td>
            <td><span class="severity medium">${item.risk_band || 'monitor'}</span></td>
            <td>${formatMetric(item.relative_risk_score)}</td>
            <td>${item.matched_cells}</td>
        </tr>
    `).join('');
}

function renderInvestorCharts() {
    if (pressureChartCtx) {
        drawBarChart(
            pressureChartCtx,
            (investorState.summary?.top_features || []).slice(0, 6).map(item => ({
                label: truncateLabel(item.feature),
                value: Number(item.importance || 0)
            })),
            '#ff8c42'
        );
    }

    if (tempChartCtx) {
        const monthly = investorState.summary?.metrics?.per_month || {};
        const chartData = Object.keys(monthly)
            .sort((a, b) => Number(a) - Number(b))
            .map((month) => ({
                label: month.padStart(2, '0'),
                value: Number(monthly[month].precision_top_decile || 0)
            }));
        drawBarChart(tempChartCtx, chartData, '#5bc0eb');
    }
}

function drawBarChart(ctx, data, color) {
    clearCanvas(ctx);
    if (!data || !data.length) return;
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    const maxValue = Math.max(...data.map(item => item.value), 1);
    const barWidth = width / data.length;

    ctx.fillStyle = '#93a0ba';
    ctx.font = '11px HK Grotesk, Arial, sans-serif';

    data.forEach((item, index) => {
        const barHeight = (item.value / maxValue) * (height - 30);
        const x = index * barWidth + 10;
        const y = height - barHeight - 18;
        ctx.fillStyle = color;
        ctx.fillRect(x, y, Math.max(barWidth - 18, 8), barHeight);
        ctx.fillStyle = '#93a0ba';
        ctx.fillText(item.label, x, height - 4);
    });
}

function renderInvestorMap() {
    const topCells = investorState.monthly?.top_cells || [];
    if (!window.esriConstructors || !graphicsLayer) {
        if (isInvestorMode()) {
            setTimeout(() => renderInvestorMap(), 800);
        }
        return;
    }

    const { Graphic, Point, SimpleMarkerSymbol } = window.esriConstructors;
    graphicsLayer.removeAll();

    topCells.forEach((cell) => {
        const band = cell.risk_band || 'monitor';
        const color = band === 'extreme'
            ? [255, 92, 92, 0.8]
            : band === 'high'
                ? [255, 148, 77, 0.75]
                : band === 'elevated'
                    ? [255, 204, 102, 0.72]
                    : [120, 176, 255, 0.65];

        const point = new Point({
            longitude: cell.lon_center,
            latitude: cell.lat_center
        });

        const symbol = new SimpleMarkerSymbol({
            style: 'diamond',
            color,
            size: `${Math.max(10, Math.round((cell.relative_risk_score || 40) / 4))}px`,
            outline: {
                color: [255, 255, 255, 0.9],
                width: 1
            }
        });

        const graphic = new Graphic({
            geometry: point,
            symbol,
            attributes: {
                type: 'wildfire-signal',
                risk_band: band,
                relative_risk_score: cell.relative_risk_score,
                predicted_probability: cell.predicted_probability
            }
        });
        graphicsLayer.add(graphic);
    });
}

function formatMetric(value) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) {
        return 'n/a';
    }
    const numeric = Number(value);
    if (numeric >= 1 && numeric <= 100 && Number.isInteger(numeric)) {
        return String(numeric);
    }
    return numeric.toFixed(3);
}

function truncateLabel(value) {
    return value.length > 14 ? `${value.slice(0, 14)}…` : value;
}
