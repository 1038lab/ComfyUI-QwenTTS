import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    QwenTTS: { nodeColor: "#151c19", nodeBgColor: "#3e5249", width: 300 },
    QwenTTSAdv: { nodeColor: "#151c19", nodeBgColor: "#24302b", width: 300 },
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
    "AILab_Qwen3TTSVoicesLibrary": "Tools",
    "AILab_Qwen3TTSLoadVoice": "Tools",
    "AILab_Qwen3TTSWhisperSTT": "Tools",
    "AILab_Qwen3TTSVoiceInstruct": "Tools",
    "AILab_Qwen3TTSVoiceInstructZH": "Tools",
    "AILab_AudioDuration": "Tools",
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
    name: "QwenTTS.appearance",

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
