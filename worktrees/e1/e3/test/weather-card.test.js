import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const SRC = pathToFileURL('src/weather-card.js');
const MOD = await import(SRC.href);

/** Shanghai payload (data/shanghai.json) */
const SHANGHAI = JSON.parse(
  readFileSync('data/shanghai.json', 'utf8')
);

describe('unitOf', () => {
  it('defaults to °C', () => {
    assert.equal(MOD.unitOf({ temp: 20 }), '°C');
  });
  it('reads unit field', () => {
    assert.equal(MOD.unitOf({ temp: 20, unit: 'F' }), '°F');
  });
  it('keeps ° symbol', () => {
    assert.equal(MOD.unitOf({ temp: 20, unit: '°F' }), '°F');
  });
});

describe('resolveCondition', () => {
  it('matches icon name', () => {
    assert.deepEqual(MOD.resolveCondition({ icon: 'rain' }), { code: 'rain', emoji: '🌧️', label: 'Rainy' });
  });
  it('matches keyword', () => {
    assert.equal(MOD.resolveCondition({ condition: 'light rain' }).code, 'rain');
  });
  it('falls back to cloudy', () => {
    assert.equal(MOD.resolveCondition({ condition: '???unknown' }).code, 'cloudy');
  });
});

describe('renderWeatherCard', () => {
  it('escapes header text', () => {
    const html = MOD.renderWeatherCard({ city: '<b>', temp: 20, condition: 'clear' });
    assert.ok(html.includes('&lt;b&gt;'));
  });
  it('includes temp + unit', () => {
    const html = MOD.renderWeatherCard({ city: 'Tokyo', temp: 22, unit: '°C', condition: 'clear' });
    assert.ok(html.includes('22°C'));
  });
  it('renders stats rows', () => {
    const html = MOD.renderWeatherCard({
      city: 'Tokyo', temp: 22, condition: 'clear',
      feelsLike: 20, humidity: 65, windKmh: 12, windDirection: 'NW', high: 25, low: 17,
    });
    assert.ok(html.includes('Feels like') && html.includes('Humidity') && html.includes('65%'));
  });
});

describe('mapShanghaiData', () => {
  it('maps weather_code → condition', () => {
    const city = MOD.mapShanghaiData(SHANGHAI);
    assert.equal(city.condition.code, 'drizzle');
    assert.equal(city.condition.emoji, '🌦️');
  });
  it('reads current temperature + unit', () => {
    const city = MOD.mapShanghaiData(SHANGHAI);
    assert.equal(city.temp, 20.7);
    assert.equal(city.unit, '°C');
  });
  it('maps wind speed m/s → km/h', () => {
    const city = MOD.mapShanghaiData(SHANGHAI);
    assert.ok(Math.abs(city.windKmh - 15.2 * 3.6) < 1e-9);
  });
  it('builds the 3-day daily trend', () => {
    const city = MOD.mapShanghaiData(SHANGHAI);
    assert.equal(city.daily.length, 3);
    assert.equal(city.daily[0].label, '中阵雨');
    assert.equal(city.daily[0].condition.code, 'rain');
    assert.equal(city.daily[0].high, 20.9);
    assert.equal(city.daily[0].low, 18.1);
  });
  it('renders into a friendly card', () => {
    const city = MOD.mapShanghaiData(SHANGHAI);
    const html = MOD.renderWeatherCard(city);
    assert.ok(html.includes('Shanghai'));
    assert.ok(html.includes('毛毛雨'));
    assert.ok(html.includes('20.7°C'));
  });
});
