/**
 * weather-card.js — dependency-free module that renders weather data into
 * a friendly card UI.
 *
 * Public API
 * ----------
 *  renderWeatherCard(city)              -> string   (HTML for one card)
 *  renderWeatherGrid(cities)            -> string   (HTML for a grid of cards)
 *  renderWeatherGridToElement(el, cities) -> Element|null
 *  injectStyles()                       -> void     (injects the card CSS, browser only)
 *
 *  Data normalization (data-fetching egg → card schema)
 *  ---------------------------------------------------
 *  mapShanghaiData(raw)                 -> object  (Open-Meteo-style raw data → card object,
 *                                                   includes a `daily` array for the 3-day trend)
 *
 *  `city` shape (missing fields are gracefully omitted):
 *   {
 *     city: 'Tokyo',                     // required
 *     country: 'JP',                     // optional
 *     temp: 22,                          // required
 *     unit: '°C',                        // optional ('C' | 'F' | '°C' | '°F'), default '°C'
 *     feelsLike: 20,                     // optional
 *     condition: 'partly-cloudy',        // code, icon name, or free text ('light rain'…)
 *     icon: 'cloudy',                    // alternative to `condition`
 *     description: 'Mostly sunny',       // optional human-readable description
 *     humidity: 65,                      // optional, %
 *     windKmh: 12,                       // optional
 *     windDirection: 'NW',               // optional
 *     high: 25, low: 17,                 // optional
 *     time: '14:18',                     // optional display time
 *     hourly: [{ time: '15:00', temp: 22, condition: 'cloudy' }]
 *     daily:  [{ date, label, condition, description, high, low }]   // optional, 3-day trend
 *   }
 */

const ICONS = {
  clear:           { emoji: '☀️', label: 'Sunny' },
  'clear-night':   { emoji: '🌙', label: 'Clear night' },
  'partly-cloudy': { emoji: '🌤️', label: 'Partly cloudy' },
  cloudy:          { emoji: '☁️', label: 'Cloudy' },
  overcast:        { emoji: '☁️', label: 'Overcast' },
  fog:             { emoji: '🌫️', label: 'Foggy' },
  drizzle:         { emoji: '🌦️', label: 'Drizzle' },
  rain:            { emoji: '🌧️', label: 'Rainy' },
  'heavy-rain':    { emoji: '🌧️', label: 'Heavy rain' },
  thunderstorm:    { emoji: '⛈️', label: 'Thunderstorm' },
  snow:            { emoji: '❄️', label: 'Snowy' },
  sleet:           { emoji: '🌨️', label: 'Sleet' },
};

const GRADIENTS = {
  clear:           'linear-gradient(135deg, #ffd166, #f77f00)',
  'clear-night':   'linear-gradient(135deg, #1e3a5f, #0d1b2a)',
  'partly-cloudy': 'linear-gradient(135deg, #90caf9, #4fc3f7)',
  cloudy:          'linear-gradient(135deg, #b0bec5, #78909c)',
  overcast:        'linear-gradient(135deg, #90a4ae, #546e7a)',
  fog:             'linear-gradient(135deg, #eceff1, #b0bec5)',
  drizzle:         'linear-gradient(135deg, #78909c, #546e7a)',
  rain:            'linear-gradient(135deg, #546e7a, #263238)',
  'heavy-rain':    'linear-gradient(135deg, #37474f, #1a2228)',
  thunderstorm:    'linear-gradient(135deg, #455a64, #1a2228)',
  snow:            'linear-gradient(135deg, #e0f7fa, #b2dfdb)',
  sleet:           'linear-gradient(135deg, #cfd8dc, #90a4ae)',
};

const LIGHT = ['clear', 'partly-cloudy', 'cloudy', 'overcast', 'fog', 'drizzle', 'snow', 'sleet'];

const FALLBACK = { emoji: '☁️', label: 'Weather' };

const KEYWORD_RULES = [
  { re: /thunder|lightning|storm/, name: { code: 'thunderstorm', emoji: '⛈️', label: 'Thunderstorm' } },
  { re: /snow|sleet/,              name: { code: 'snow', ...ICONS.snow } },
  { re: /rain|drizzle/,           name: { code: 'rain', ...ICONS.rain } },
  { re: /fog|mist|haze/,          name: { code: 'fog', ...ICONS.fog } },
  { re: /cloud|overcast/,         name: { code: 'cloudy', ...ICONS.cloudy } },
  { re: /clear|sunny|sun/,        name: { code: 'clear', ...ICONS.clear } },
];

/**
 * Open-Meteo WMO weather codes → card condition codes, used to normalize
 * the structured data returned by the data-fetching egg.
 */
const WMO_CODE_MAP = {
  0: 'clear', 1: 'partly-cloudy', 2: 'partly-cloudy', 3: 'overcast',
  45: 'fog', 48: 'fog',
  51: 'drizzle', 53: 'drizzle', 55: 'drizzle',
  56: 'sleet', 57: 'sleet',
  61: 'rain', 63: 'rain', 65: 'rain',
  66: 'sleet', 67: 'sleet',
  71: 'snow', 73: 'snow', 75: 'snow', 77: 'snow',
  80: 'rain', 81: 'rain', 82: 'heavy-rain',
  85: 'snow', 86: 'snow',
  95: 'thunderstorm', 96: 'thunderstorm', 99: 'thunderstorm',
};

/** Chinese labels for WMO codes (used by the Shanghai card). */
const WMO_ZH = {
  0: '晴', 1: '晴', 2: '少云', 3: '多云', 45: '雾', 48: '雾',
  51: '毛毛雨', 53: '毛毛雨', 55: '毛毛雨', 56: '冻雨', 57: '冻雨',
  61: '小雨', 63: '中雨', 65: '大雨', 66: '冻雨', 67: '冻雨',
  71: '小雪', 73: '中雪', 75: '大雪', 77: '雪粒',
  80: '阵雨', 81: '中阵雨', 82: '大雨', 85: '小雪花', 86: '大雪花',
  95: '雷暴', 96: '雷暴', 99: '雷暴',
};

function codeToCondition(code) {
  const key = WMO_CODE_MAP[code];
  return key && ICONS[key] ? { code: key, ...ICONS[key] } : null;
}

function round1(n) {
  return Math.round(n * 10) / 10;
}

function extractTime(observedAt) {
  const m = String(observedAt ?? '').match(/T(\d{2}:\d{2})/);
  return m ? m[1] : null;
}

const WEEKDAY_ZH = ['一', '二', '三', '四', '五', '六', '日'];

function weekdayOf(dateStr) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(dateStr ?? ''));
  if (!m) return '';
  return WEEKDAY_ZH[new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])).getDay()];
}
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

export function unitOf(city) {
  const u = String(city.unit ?? city.tempUnit ?? 'C').replace(/°/g, '').toUpperCase();
  return u === 'F' ? '°F' : '°C';
}

export function resolveCondition(city) {
  const raw = String(
    city.condition ?? city.icon ?? city.weather?.condition ?? city.weather?.icon ?? 'clear'
  ).toLowerCase().replace(/\s+/g, '-');
  if (ICONS[raw]) return { code: raw, ...ICONS[raw] };
  for (const rule of KEYWORD_RULES) {
    if (rule.re.test(raw)) return { code: rule.name.code, ...rule.name };
  }
  return { code: 'cloudy', ...FALLBACK };
}

function themeFor(code) {
  const light = LIGHT.includes(code);
  return {
    text: light ? '#1a202c' : '#f7fafc',
    chip: light ? 'rgba(255, 255, 255, 0.30)' : 'rgba(255, 255, 255, 0.14)',
  };
}

function statRows(city) {
  const rows = [
    { label: 'Feels like', value: city.feelsLike != null ? `${esc(city.feelsLike)}${unitOf(city)}` : null },
    { label: 'Humidity',   value: city.humidity   != null ? `${esc(city.humidity)}%` : null },
    { label: 'Wind',       value: city.windKmh    != null
      ? `${esc(city.windKmh)} km/h${city.windDirection ? ' ' + esc(city.windDirection) : ''}`
      : null },
    { label: 'Today',      value: city.high != null && city.low != null
      ? `${esc(city.high)}° / ${esc(city.low)}°` : null },
  ];
  return rows
    .filter((r) => r.value != null)
    .map((r) => `<div class="wc-stat"><span class="wc-stat-label">${r.label}</span><span class="wc-stat-value">${r.value}</span></div>`)
    .join('');
}

function hourlyRow(hourly, unit) {
  if (!Array.isArray(hourly) || hourly.length === 0) return '';
  const items = hourly
    .map((h) => {
      const cond = resolveCondition(h);
      const t = h.temp != null ? `${esc(h.temp)}${unit}` : '';
      return `<div class="wc-hour"><span class="wc-hour-time">${esc(h.time ?? '')}</span><span class="wc-hour-icon" aria-hidden="true">${cond.emoji}</span><span class="wc-hour-temp">${t}</span></div>`;
    })
    .join('');
  return `<div class="wc-hourly" aria-label="Hourly forecast">${items}</div>`;
}

export function renderWeatherCard(city) {
  const unit = unitOf(city);
  const cond = resolveCondition(city);
  const label = esc(city.description ?? cond.label);
  const theme = themeFor(cond.code);
  const gradient = GRADIENTS[cond.code] ?? GRADIENTS.cloudy;

  const sub = city.country != null
    ? `<span class="wc-sub">${esc(city.country)}</span>`
    : '';
  const time = city.time != null ? `<span class="wc-time">${esc(city.time)}</span>` : '';

  const header = `<div class="wc-header"><div class="wc-location"><span class="wc-city">${esc(city.city)}</span>${sub}</div>${time}</div>`;

  const main = `<div class="wc-main"><span class="wc-icon" aria-hidden="true">${cond.emoji}</span><div class="wc-temp"><span class="wc-temp-value">${esc(city.temp ?? '–')}</span><span class="wc-temp-unit">${unit}</span></div><p class="wc-condition">${label}</p></div>`;

  const stats = statRows(city);
  const hourlyHtml = hourlyRow(city.hourly, unit);

  return `<article class="wc" style="--wc-bg:${gradient}; --wc-text:${theme.text}; --wc-chip:${theme.chip}">${header}${main}${stats ? `<div class="wc-stats">${stats}</div>` : ''}${hourlyHtml}</article>`;
}

export function renderWeatherGrid(cities) {
  const list = Array.isArray(cities) ? cities : [cities];
  return `<div class="wc-grid" role="list">${list.map(renderWeatherCard).join('')}</div>`;
}

export function renderWeatherGridToElement(container, cities) {
  injectStyles();
  if (container) container.innerHTML = renderWeatherGrid(cities);
  return container;
}

const CARD_CSS = `
.wc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
}
.wc {
  position: relative;
  border-radius: 20px;
  background: var(--wc-bg);
  color: var(--wc-text, #1a202c);
  padding: 16px 16px 14px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.wc:hover { transform: translateY(-3px); box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22); }
.wc-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
.wc-location { display: flex; flex-direction: column; gap: 2px; }
.wc-city { font-size: 1.05rem; font-weight: 700; }
.wc-sub { font-size: 0.8rem; opacity: 0.75; }
.wc-time { font-size: 0.8rem; opacity: 0.8; white-space: nowrap; }
.wc-main { display: flex; align-items: center; gap: 14px; margin: 10px 0 14px; }
.wc-icon { font-size: 3rem; line-height: 1; }
.wc-temp { font-size: 2.9rem; font-weight: 800; line-height: 1; }
.wc-temp-unit { font-size: 1.05rem; font-weight: 600; opacity: 0.85; margin-left: 2px; }
.wc-condition { margin: 0; font-size: 0.95rem; opacity: 0.9; }
.wc-stats {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  padding: 10px;
  border-radius: 12px;
  background: var(--wc-chip);
}
.wc-stat { display: flex; flex-direction: column; gap: 2px; }
.wc-stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8; }
.wc-stat-value { font-size: 0.95rem; font-weight: 700; }
.wc-hourly { display: flex; gap: 6px; margin-top: 12px; overflow-x: auto; }
.wc-hour {
  flex: 1;
  min-width: 54px;
  text-align: center;
  border-radius: 10px;
  padding: 6px 4px;
  background: var(--wc-chip);
  font-size: 0.78rem;
}
.wc-hour-time { display: block; opacity: 0.8; }
.wc-hour-icon { display: block; font-size: 1.15rem; margin: 2px 0; }
.wc-hour-temp { display: block; font-weight: 700; }
@media (prefers-reduced-motion: reduce) { .wc { transition: none; } }
@media (max-width: 420px) {
  .wc-grid { grid-template-columns: 1fr; }
  .wc-temp { font-size: 2.4rem; }
}
`;

export function injectStyles() {
  if (typeof document === 'undefined') return;
  if (document.getElementById('weather-card-style')) return;
  const style = document.createElement('style');
  style.id = 'weather-card-style';
  style.textContent = CARD_CSS;
  document.head.appendChild(style);
}

