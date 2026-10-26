const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const source = fs.readFileSync(__dirname + '/static/i18n.js', 'utf8');
const html = fs.readFileSync(__dirname + '/static/index.html', 'utf8');
function environment(storage = new Map()) {
  const context = {localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)}};
  vm.createContext(context);
  vm.runInContext(source, context);
  return {i18n: context.DropletI18n, context};
}
const storage = new Map();
const {i18n} = environment(storage);
assert.equal(i18n.getLanguage(), 'zh-CN');
i18n.setLanguage('en');
assert.equal(i18n.t('图像与标注'), 'Images & annotations');
assert.equal(i18n.t('已选 {count} / {total} 张', {count: 3, total: 85}), '3 / 85 selected');
assert.equal(i18n.t('{name}：{count} 张', {name: '图像与标注.png', count: 2}), '图像与标注.png: 2 images');
assert.equal(i18n.translateError('同一实验批次不能分到不同集合：回收区.png'), 'An experiment batch cannot span dataset splits: 回收区.png');
assert.equal(i18n.translateError('单液滴分类：only cell · 97.2%（需人工审核）'), 'Single-droplet classification: only cell · 97.2% (review required)');
assert.equal(i18n.t('永久删除'), 'DELETE PERMANENTLY');
assert.equal(i18n.t('清空所有项目'), 'DELETE ALL PROJECTS');
assert.equal(environment(storage).i18n.getLanguage(), 'en');
const keys = new Set();
for (const match of html.matchAll(/data-i18n(?:-(?:placeholder|aria-label|alt|title))?="([^"]+)"/g)) keys.add(match[1].replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, '&'));
for (const match of html.matchAll(/\bt\((['"])([^'"\n]*)\1/g)) keys.add(match[2]);
for (const match of html.matchAll(/uiText\('[^']+','([^']+)'/g)) keys.add(match[1]);
for (const key of keys) assert(Object.hasOwn(i18n.english, key), 'Missing English translation: ' + key);
function placeholders(value) { return Array.from(value.matchAll(/\{(\w+)\}/g), m => m[1]).sort(); }
for (const [zh, en] of Object.entries(i18n.english)) assert.deepEqual(placeholders(zh), placeholders(en), 'Mismatched placeholders: ' + zh);
i18n.setLanguage('zh-CN');
assert.equal(i18n.t('已选 {count} / {total} 张', {count: 3, total: 85}), '已选 3 / 85 张');
i18n.setLanguage('unsupported');
assert.equal(i18n.getLanguage(), 'zh-CN');
const denied = {localStorage: {getItem() {throw Error('storage blocked');}, setItem() {throw Error('storage blocked');}}};
vm.createContext(denied); vm.runInContext(source, denied);
denied.DropletI18n.setLanguage('en');
assert.equal(denied.DropletI18n.t('使用帮助'), 'Help');
console.log(`Bilingual catalog coverage (${keys.size} UI keys), placeholders, persisted preference, storage fallback, error translation and unchanged user filenames passed`);
