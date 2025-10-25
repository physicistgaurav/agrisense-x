import anthropic

# Initialize Claude client
client = anthropic.Anthropic(
    api_key="YOUR_API_KEY")


def generate_advice(crop, disease, confidence, language="english", context=None):
    """
    Generate comprehensive agricultural advice using Claude

    Args:
        crop: Crop type
        disease: Detected disease
        confidence: Model confidence score
        language: Target language for response
        context: Optional context from previous conversations
    """

    prompt = f"""You are an expert agricultural pathologist and advisor with decades of field experience.

*Diagnosis Details:**
- Crop: {crop}
- Disease Detected: {disease}
- AI Confidence: {confidence:.1%}

Please provide a comprehensive, actionable advisory in {language}. Structure your response as follows:

1. **Disease Overview** (2-3 sentences)
   - What is this disease?
   - How serious is it?

2. **Visual Symptoms** (bullet points)
   - What farmers should look for
   - Disease progression stages

3. **Root Causes**
   - Primary causes
   - Environmental factors
   - Transmission methods

4. **Immediate Action Plan** (next 24-48 hours)
   - Critical steps numbered 1, 2, 3...
   - What to avoid

5. **Treatment Options**
   - Organic methods
   - Chemical treatments (with safety notes)
   - Estimated costs

6. **Prevention Strategy**
   - Long-term practices
   - Crop rotation suggestions
   - Resistant varieties

7. **Expected Timeline**
   - Treatment duration
   - Recovery indicators
   - When to see improvement

8. **Economic Impact**
   - Potential yield loss if untreated
   - Treatment cost vs. loss prevention

{"Remember: This is a low-confidence detection. Recommend manual expert verification." if confidence < 0.6 else ""}

Be practical, empathetic, and consider that farmers may have limited resources. Use simple language while being scientifically accurate."""

    messages = [{"role": "user", "content": prompt}]

    # Add context from previous conversation if available
    if context:
        messages.insert(0, {"role": "assistant", "content": context})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=messages
    )

    return response.content[0].text


def chat_with_advisor(user_question, disease_context, chat_history=None):
    """
    Interactive chat feature for follow-up questions

    Args:
        user_question: Farmer's question
        disease_context: Current disease detection context
        chat_history: Previous conversation for continuity
    """

    system_prompt = f"""You are an agricultural advisor chatting with a farmer about their crop disease.

Current Diagnosis Context:
- Disease: {disease_context['disease']}
- Crop: {disease_context['crop']}
- Confidence: {disease_context['confidence']:.1%}

Guidelines:
- Answer in a warm, conversational tone
- Be practical and consider resource constraints
- If the question is outside plant disease, gently redirect
- Use emojis occasionally to be friendly
- Keep responses concise (3-5 sentences unless more detail needed)
- If you don't know, be honest and suggest consulting a local expert"""

    messages = [
        {"role": "user", "content": f"{system_prompt}\n\nFarmer's question: {user_question}"}
    ]

    # Add chat history for context
    if chat_history:
        for msg in chat_history[-6:]:  # Last 3 exchanges
            messages.insert(0, msg)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=messages
    )

    return response.content[0].text


def generate_multilingual_summary(disease, language):
    """
    Generate quick disease summary in local languages
    Supports: English, Hindi, Nepali
    """

    language_map = {
        "hindi": "Hindi (हिंदी)",
        "nepali": "Nepali (नेपाली)",
        "english": "English"
    }

    prompt = f"""Provide a brief 3-sentence summary about {disease} in {language_map.get(language, language)}.
    
Focus on:
1. What it is
2. Main symptom
3. Quick action to take

Keep it simple for farmers with basic education."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def compare_with_similar_diseases(disease, confidence):
    """
    Generate differential diagnosis for low-confidence predictions
    """

    prompt = f"""The AI detected "{disease}" with {confidence:.1%} confidence.

As an expert, provide a differential diagnosis:

1. List 3 other diseases with similar visual symptoms
2. For each, explain key distinguishing features
3. Suggest specific tests or observations to confirm diagnosis

Be concise but scientific."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def generate_seasonal_tips(crop, current_month):
    """
    Generate season-specific prevention tips
    """

    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    month_name = months[current_month - 1]

    prompt = f"""For {crop} crops in {month_name}:

Provide 5 specific preventive measures farmers should take this month to avoid common diseases.

Format as numbered list with brief explanations."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
