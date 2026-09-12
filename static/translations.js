(function () {
    'use strict';

    var TRANSLATIONS = {
        en: {
            appTitle: "BuddyAD",
            videoFile: "Video File",
            selectFile: "Select File",
            llm: "LLM",
            geminiKey: "Gemini API Key",
            test: "Test",
            apiBaseUrl: "API Base URL",
            modelName: "Model Name",
            apiKey: "API Key",
            enableGap: "Enable dialogue gap detection",
            useContext: "Using multi-shot context to improve AD continuity",
            videoType: "Video Type",
            optMovie: "Movie",
            optTv: "TV Series",
            optStage: "Stage Performance",
            optCustom: "Custom",
            customPrompt: "Custom prompt:",
            customPromptNote: "This replaces the first opening sentence. The \"Where different shot numbers are labelled...\" sentence and the rest of the description template are kept as-is.",
            start: "Start",
            processing: "Processing...",
            initializing: "Initializing...",
            completed: "Complete!",
            finishedIssues: "Finished with issues",
            dlCsv: "Download Main CSV",
            dlStage1: "Download Stage 1",
            dlStage2: "Download Stage 2",
            dlVtt: "Download Subtitles (VTT)",
            dlPartial: "Download partial results (Stage 1)",
            noFiles: "No downloadable files were produced for this run.",
            another: "Process Another Video",
            errorTitle: "Error",
            busy: "Another task is already processing. Wait for it to finish.",
            pleaseSelect: "Please select a video file",
            pleaseKey: "Please enter your API key - without it the output CSVs will be empty",
            testing: "Testing...",
            connectionLost: "Connection lost",
            backendNotReady: "[ERROR] Backend not ready yet - try again in a moment",
            okPrefix: "[OK] ",
            errPrefix: "[ERROR] ",
            reqFailed: "[ERROR] Request failed: ",
            tokenUsage: "Token usage",
            usageStage1: "Stage 1",
            usageStage2: "Stage 2",
            usageTotal: "Total",
            usageCalls: "calls",
            usageInput: "input",
            usageCached: "cached",
            usageOutput: "output",
            switchToZh: "中文",
            switchToEn: "EN"
        },
        zh: {
            appTitle: "口述影像助理",
            videoFile: "影片檔案",
            selectFile: "選擇檔案",
            llm: "語言模型",
            geminiKey: "Gemini 金鑰",
            test: "測試",
            apiBaseUrl: "API 位址",
            modelName: "模型名稱",
            apiKey: "API 金鑰",
            enableGap: "啟用對話間隙偵測",
            useContext: "利用多鏡頭上下文來改善 AD 的連續性",
            videoType: "影片類型",
            optMovie: "電影",
            optTv: "影集",
            optStage: "舞台表演",
            optCustom: "自訂",
            customPrompt: "自訂提示：",
            customPromptNote: "此文字會取代開場第一句；「Where different shot numbers are labelled...」及其餘描述範本維持不變。",
            start: "開始",
            processing: "處理中…",
            initializing: "初始化中…",
            completed: "完成！",
            finishedIssues: "有問題地完成",
            dlCsv: "下載主 CSV",
            dlStage1: "下載第一階段",
            dlStage2: "下載第二階段",
            dlVtt: "下載字幕 (VTT)",
            dlPartial: "下載部分結果（第一階段）",
            noFiles: "本次未產生可下載的檔案。",
            another: "處理其他影片",
            errorTitle: "錯誤",
            busy: "已有其他任務正在處理，請等它完成。",
            pleaseSelect: "請選擇影片檔案",
            pleaseKey: "請輸入 API 金鑰，否則輸出的 CSV 會是空的",
            testing: "測試中…",
            connectionLost: "連線中斷",
            backendNotReady: "[錯誤] 後端尚未就緒，請稍後再試",
            okPrefix: "[成功] ",
            errPrefix: "[錯誤] ",
            reqFailed: "[錯誤] 請求失敗：",
            tokenUsage: "Token 用量",
            usageStage1: "第一階段",
            usageStage2: "第二階段",
            usageTotal: "總計",
            usageCalls: "請求次數",
            usageInput: "輸入",
            usageCached: "快取",
            usageOutput: "輸出",
            switchToZh: "中文",
            switchToEn: "EN"
        }
    };

    var I18N_KEY = 'uiLang';

    function stored() {
        try { return localStorage.getItem(I18N_KEY) || 'auto'; } catch (e) { return 'auto'; }
    }

    function resolve() {
        var s = stored();
        if (s === 'en' || s === 'zh') return s;
        var nav = (navigator.language || '').toLowerCase();
        return nav.indexOf('zh') === 0 ? 'zh' : 'en';
    }

    function current() {
        var lang = resolve();
        return {
            lang: lang,
            t: function (key) {
                var table = TRANSLATIONS[lang];
                return (table && table[key] !== undefined) ? table[key] : key;
            }
        };
    }

    function convertInline() {
        // data-i18n="key"        -> element text
        // data-i18n-attr="k=key" -> element attribute (e.g. placeholder)
        var els = document.querySelectorAll('[data-i18n]');
        var cur = current();
        for (var i = 0; i < els.length; i++) {
            var el = els[i];
            if (el.tagName === 'TITLE') continue;
            el.textContent = cur.t(el.getAttribute('data-i18n'));
        }
        var attrEls = document.querySelectorAll('[data-i18n-attr]');
        for (var j = 0; j < attrEls.length; j++) {
            var ae = attrEls[j];
            var spec = (ae.getAttribute('data-i18n-attr') || '').split(',');
            for (var k = 0; k < spec.length; k++) {
                var kv = spec[k].split('=');
                if (kv.length === 2) ae.setAttribute(kv[0].trim(), cur.t(kv[1].trim()));
            }
        }
        var toggle = document.getElementById('langToggle');
        if (toggle) toggle.textContent = cur.lang === 'zh' ? cur.t('switchToEn') : cur.t('switchToZh');
        document.title = cur.t('appTitle');
        try { document.documentElement.lang = cur.lang === 'zh' ? 'zh-Hant' : 'en'; } catch (e) {}
    }

    function apply() {
        convertInline();
        if (typeof window.onI18nChanged === 'function') window.onI18nChanged();
    }

    function switchLang() {
        var cur = current();
        var next = cur.lang === 'zh' ? 'en' : 'zh';
        try { localStorage.setItem(I18N_KEY, next); } catch (e) {}
        apply();
    }

    window.I18N = {
        init: apply,
        t: function (key) { return current().t(key); },
        lang: function () { return current().lang; },
        switch: switchLang
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', apply);
    } else {
        apply();
    }
})();