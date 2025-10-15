# 🌱 AgriSense-X: Explainable Multimodal Crop Disease Detection & Advisory System

> An AI-powered agricultural assistant combining Computer Vision and Large Language Models to provide intelligent crop disease diagnosis with actionable treatment advice.

## 🎯 Project Highlights

### What Makes This Special?

1. **Explainable AI** - Not just predictions, but visual explanations using GradCAM showing exactly where the AI is looking
2. **Conversational Expert** - Chat with Claude AI for follow-up questions about treatments, costs, and concerns
3. **Multilingual Support** - Advice in English, Hindi, Nepali, and Spanish for broader farmer accessibility
4. **Treatment Timeline** - Actionable checklists and progress tracking
6. **Low-Confidence Alerts** - Differential diagnosis for uncertain predictions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AgriSense-X System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Image       │───▶│  MobileNetV2 │───▶│  Disease     │ │
│  │   Upload      │    │  Classifier  │    │  Prediction  │ │
│  └───────────────┘    └──────────────┘    └──────────────┘ │
│                              │                      │         │
│                              ▼                      ▼         │
│                       ┌──────────────┐    ┌──────────────┐  │
│                       │   GradCAM    │    │  Claude AI   │  │
│                       │ Explainability│   │   Advisory   │  │
│                       └──────────────┘    └──────────────┘  │
│                              │                      │         │
│                              ▼                      ▼         │
│                       ┌─────────────────────────────────┐   │
│                       │    Streamlit Web Interface      │   │
│                       │  • Visual Explanations          │   │
│                       │  • Conversational Chat          │   │
│                       │  • Multilingual Support         │   │
│                       │  • Analytics Dashboard          │   │
│                       └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```