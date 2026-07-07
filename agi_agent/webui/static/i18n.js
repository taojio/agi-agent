const I18N = (function() {
    let currentLang = 'zh';
    let translations = {};
    const supportedLangs = ['zh', 'en'];

    async function loadLang(lang) {
        if (!supportedLangs.includes(lang)) lang = 'zh';
        
        try {
            const response = await fetch(`/static/locales/${lang}.json`);
            translations = await response.json();
            currentLang = lang;
            localStorage.setItem('agi_agent_lang', lang);
            document.documentElement.lang = lang;
            return true;
        } catch (error) {
            console.error(`Failed to load language ${lang}:`, error);
            return false;
        }
    }

    function get(key, params = {}) {
        const keys = key.split('.');
        let value = translations;
        
        for (const k of keys) {
            if (value && typeof value === 'object' && k in value) {
                value = value[k];
            } else {
                console.warn(`Missing translation key: ${key}`);
                return key;
            }
        }
        
        if (typeof value === 'string') {
            for (const [paramKey, paramValue] of Object.entries(params)) {
                value = value.replace(`{${paramKey}}`, paramValue);
            }
        }
        
        return value;
    }

    function getCurrentLang() {
        return currentLang;
    }

    function getSupportedLangs() {
        return supportedLangs;
    }

    async function init() {
        const savedLang = localStorage.getItem('agi_agent_lang');
        const browserLang = navigator.language || navigator.userLanguage;
        const defaultLang = browserLang.startsWith('zh') ? 'zh' : 'en';
        const langToLoad = savedLang || defaultLang;
        await loadLang(langToLoad);
    }

    return {
        loadLang,
        get,
        getCurrentLang,
        getSupportedLangs,
        init
    };
})();