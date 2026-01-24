import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    QwenTTS: { nodeColor: "#4D2600", nodeBgColor: "#3e5249", width: 340 },
    QwenTTSAdv: { nodeColor: "#4D2600", nodeBgColor: "#24302b", width: 340 },
    Tools: { nodeColor: "#28403f", nodeBgColor: "#233238", width: 300 },
};

const NODE_COLORS = {
    // QwenTTS nodes
    "AILab_Qwen3TTSCustomVoice": "QwenTTS",
    "AILab_Qwen3TTSCustomVoice_Advanced": "QwenTTSAdv",
    "AILab_Qwen3TTSVoiceDesign": "QwenTTS",
    "AILab_Qwen3TTSVoiceDesign_Advanced": "QwenTTSAdv",
    "AILab_Qwen3TTSVoiceClone": "QwenTTS",
    "AILab_Qwen3TTSVoiceClone_Advanced": "QwenTTSAdv",

    // Tools
    "AILab_QwenVL_PromptLibrary": "Tools",
};

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

const ext = {
    name: "QwenVL.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);
