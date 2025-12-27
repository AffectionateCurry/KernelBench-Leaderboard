// KernelBench Leaderboard - Common Utilities

// Theme Management
function initTheme() {
    const theme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
}

function updateThemeIcon() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const icon = isDark ? '\u2600\uFE0F' : '\uD83C\uDF19';
    document.querySelectorAll('.theme-icon').forEach(el => el.textContent = icon);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon();
}

// Initialize theme toggles
function initThemeToggles() {
    document.querySelectorAll('[id^="theme-toggle"]').forEach(btn => {
        btn.addEventListener('click', toggleTheme);
    });
    updateThemeIcon();
}

// HTML Escape
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format speedup with color class
function formatSpeedup(speedup) {
    if (speedup >= 1.0) return { class: 'good', text: speedup.toFixed(2) + 'x' };
    if (speedup >= 0.8) return { class: 'warning', text: speedup.toFixed(2) + 'x' };
    return { class: 'bad', text: speedup.toFixed(2) + 'x' };
}

// Get speedup color
function getSpeedupColor(speedup) {
    if (speedup >= 1.0) return 'var(--success)';
    if (speedup >= 0.8) return 'var(--warning)';
    return 'var(--danger)';
}

// Clean kernel name from filename
function cleanKernelName(filename) {
    let name = filename.replace(/\.py$/, '');
    name = name.replace(/_/g, ' ').trim();
    name = name.replace(/^\d+\s*/, '');
    return name;
}

// Parse URL query params
function getQueryParams() {
    const params = new URLSearchParams(window.location.search);
    const result = {};
    for (const [key, value] of params) {
        result[key] = value;
    }
    return result;
}

// Model colors for charts
const MODEL_COLORS = [
    '#58a6ff', '#f0883e', '#a371f7', '#3fb950',
    '#f778ba', '#56d4dd', '#db6d28', '#7ee787'
];

function getModelColor(index) {
    return MODEL_COLORS[index % MODEL_COLORS.length];
}

// Level colors
const LEVEL_COLORS = {
    1: '#58a6ff',
    2: '#a371f7',
    3: '#f778ba'
};

// Initialize common functionality
document.addEventListener('DOMContentLoaded', () => {
    initThemeToggles();
});

// Pre-initialize theme before DOM loads
initTheme();
