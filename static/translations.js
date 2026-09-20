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
            modelList: "Model list",
            modelListTitle: "Available Models",
            apply: "Apply",
            cancel: "Cancel",
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
            switchToEn: "EN",
            tabTask: "New Task",
            tabHistory: "History",
            historyEmpty: "No history yet. Process a video to create your first job.",
            openBtn: "Open",
            deleteBtn: "Delete",
            backBtn: "← Back",
            deleteConfirm: "Delete this job and all its saved frames/runs/thumbnails from disk? This cannot be undone.",
            jobStats: "shots",
            framesLabel: "frames",
            sizeLabel: "size",
            runsTitle: "Runs",
            runLabel: "Run",
            currentBadge: "current",
            running: "running",
            noDescription: "No description yet - select its shots and run 重生執行.",
            adLabel: "AD",
            selectAll: "Select all",
            clearSel: "Clear",
            selectedShots: "selected shot(s)",
            reprocessBtn: "重生執行",
            reprocessHint: "Select shots, then re-describe their AD units and re-run stage 2 with the settings below.",
            reprocessing: "Reprocessing...",
            reprocessDone: "Done. This job is now a new run; previous runs are kept.",
            sourceMissing: "Neither the source video nor the saved frames are available - reprocess is unavailable.",
            pleaseSelectShots: "Please select at least one shot.",
            kindStage1: "Stage 1",
            kindAD: "AD list",
            kindFinal: "Final CSV",
            kindVTT: "VTT",
            download: "Download",
            dataDirLabel: "Data folder:",
            openFolder: "Open folder",
            copyBtn: "Copy",
            webOpenHint: "Browsers can't open local folders - copy the path and paste it into File Explorer.",
            unitLabel: "AD unit",
            gapMode: "dialogue-gap",
            shotMode: "per-shot",
            customMode: "user range",
            rangeScope: "Describe:",
            rangeFull: "Whole video",
            rangeExtra: "Whole video, plus the segments below",
            rangeOnly: "Only the segments below",
            rangeNote: "hh:mm:ss-hh:mm:ss, separated by commas.",
            rangeEmpty: "Enter at least one time range as hh:mm:ss-hh:mm:ss.",
            rangeBadFormat: "'{v}' is not a time range - use hh:mm:ss-hh:mm:ss.",
            rangeBadTime: "'{v}' is not a valid time - use hh:mm:ss.",
            rangeOrder: "'{v}': the start time must be earlier than the end time.",
            rangeOverlap: "{a} and {b} overlap."
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
            modelList: "模型列表",
            modelListTitle: "可用模型",
            apply: "套用",
            cancel: "取消",
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
            switchToEn: "EN",
            tabTask: "新任務",
            tabHistory: "歷史",
            historyEmpty: "尚無歷史。先處理一部影片建立任務。",
            openBtn: "開啟",
            deleteBtn: "刪除",
            backBtn: "← 返回",
            deleteConfirm: "確定從磁碟刪除此任務與所有已存圖片／執行檔／縮圖？此動作無法復原。",
            jobStats: "個鏡頭",
            framesLabel: "張圖片",
            sizeLabel: "大小",
            runsTitle: "執行紀錄",
            runLabel: "執行",
            currentBadge: "目前",
            running: "處理中",
            noDescription: "尚無描述——勾選其鏡頭並按「重生執行」。",
            adLabel: "口述影像",
            selectAll: "全選",
            clearSel: "清除",
            selectedShots: "個鏡頭已選擇",
            reprocessBtn: "重生執行",
            reprocessHint: "勾選鏡頭後，以其 AD 單位重新描述並以下方設定重跑第二階段。",
            reprocessing: "重生處理中…",
            reprocessDone: "完成。本任務已成為新執行紀錄，舊執行保留。",
            sourceMissing: "原始影片與已存圖片皆不存在，無法重生執行。",
            pleaseSelectShots: "請至少選擇一個鏡頭。",
            kindStage1: "第一階段",
            kindAD: "AD 列表",
            kindFinal: "最終 CSV",
            kindVTT: "VTT",
            download: "下載",
            dataDirLabel: "資料夾：",
            openFolder: "開啟資料夾",
            copyBtn: "複製",
            webOpenHint: "瀏覽器無法直接開啟本機資料夾，請複製路徑後貼到檔案總管。",
            unitLabel: "AD 單位",
            gapMode: "對話間隙",
            shotMode: "單鏡頭",
            customMode: "自訂片段",
            rangeScope: "描述範圍：",
            rangeFull: "整段影片",
            rangeExtra: "整段影片，另外加上以下片段",
            rangeOnly: "只描述以下片段",
            rangeNote: "格式為 hh:mm:ss-hh:mm:ss，多段請以逗號分隔。",
            rangeEmpty: "請輸入至少一個時間範圍，格式為 hh:mm:ss-hh:mm:ss。",
            rangeBadFormat: "「{v}」格式不正確，應為 hh:mm:ss-hh:mm:ss。",
            rangeBadTime: "「{v}」不是有效時間，應為 hh:mm:ss。",
            rangeOrder: "「{v}」的開始時間必須早於結束時間。",
            rangeOverlap: "{a} 與 {b} 重疊。"
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