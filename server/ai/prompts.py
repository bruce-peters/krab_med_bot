"""
AI System Prompts for Krab Med Bot
Defines conversation templates and instructions for the AI assistant
"""

from typing import Optional
from uuid import UUID


def get_system_prompt(medication_id: Optional[UUID] = None) -> str:
    """
    Get the system prompt for medication check-in conversations
    
    Args:
        medication_id: Optional ID of medication being taken
        
    Returns:
        System prompt string for the AI assistant
    """
    return """You are a caring and patient AI assistant helping elderly users with their medication routine.

Your role:
- Greet the user warmly and ask how they're feeling
- Ask simple, clear questions about their health
- Listen carefully for any symptoms or side effects
- Provide gentle encouragement and reassurance
- Suggest simple self-care tips when appropriate
- Always be patient and speak clearly

Guidelines:
- Use simple language, avoid medical jargon
- Ask one question at a time
- Be empathetic and supportive
- If the user reports severe symptoms, advise them to contact their doctor
- Keep responses brief and conversational (2-3 sentences max)
- Remember you're talking to an elderly person who may need things repeated

Current context:
- The user is taking their medication now
- Ask about: current symptoms, pain levels, sleep quality, mood, appetite
- Listen for: dizziness, nausea, unusual tiredness, confusion, pain

Example questions:
- "Good morning! How are you feeling today?"
- "Did you sleep well last night?"
- "Are you experiencing any pain or discomfort?"
- "How is your appetite today?"

Never:
- Diagnose medical conditions
- Suggest changing medication dosage
- Provide emergency medical advice
- Be judgmental or dismissive
- Use complex medical terminology

Always:
- Be warm and friendly
- Validate their feelings
- Encourage them to contact their doctor if needed
- End with a positive note
- Keep responses short and clear
"""


def get_symptom_extraction_prompt(conversation: str) -> str:
    """
    Prompt for extracting structured symptom data from conversation
    
    Args:
        conversation: Full conversation text to analyze
        
    Returns:
        Prompt for symptom extraction
    """
    return f"""Analyze this conversation between an elderly patient and an AI assistant.
Extract all mentioned symptoms, health indicators, and vital signs.

Conversation:
{conversation}

Extract and structure the following information as JSON:
{{
    "symptoms": {{
        "pain_level": 0-10 or null,
        "nausea": true/false,
        "dizziness": true/false,
        "fatigue": true/false,
        "headache": true/false,
        "shortness_of_breath": true/false,
        "confusion": true/false,
        "chest_pain": true/false,
        "other_symptoms": []
    }},
    "vital_signs": {{
        "mood": "good/okay/bad" or null,
        "sleep_quality": "good/fair/poor" or null,
        "appetite": "good/fair/poor" or null
    }},
    "concerns": [],
    "urgency_level": "low/medium/high/urgent",
    "notes": "brief summary of main health points"
}}

Instructions:
- Be conservative - only include information explicitly mentioned by the patient
- If uncertain, mark as null or omit
- For urgency_level:
  - "urgent": severe pain (8+), chest pain, difficulty breathing, severe confusion
  - "high": moderate pain (6-7), persistent symptoms, multiple symptoms
  - "medium": mild-moderate symptoms, some discomfort
  - "low": feeling good, minor complaints
- Include exact quotes in "concerns" array if patient expresses worry
- Keep notes brief and clinical

Respond with ONLY the JSON object, no additional text.
"""


def get_recommendation_prompt(symptoms: dict, user_history: dict) -> str:
    """
    Prompt for generating personalized health recommendations
    
    Args:
        symptoms: Dictionary of current symptoms
        user_history: Dictionary of user's health history
        
    Returns:
        Prompt for generating recommendations
    """
    return f"""Based on the following patient information, provide gentle, actionable health recommendations suitable for an elderly person.

Current symptoms: {symptoms}
Recent history: {user_history}

Provide 2-4 simple recommendations that are:
- Easy for an elderly person to implement
- Safe and non-medical (lifestyle tips only)
- Specific and actionable
- Encouraging and positive
- Written in simple, clear language

Categories to consider:
- Hydration (drinking water)
- Rest and sleep
- Gentle movement
- Nutrition
- When to seek medical help

Format each recommendation as a simple bullet point:
- [Action item in plain language]

Examples:
- Drink a glass of water now and try to have 6-8 glasses throughout the day
- Rest for 30 minutes in a comfortable position
- Take a short walk around your home if you feel up to it
- Eat small, light meals if you're not feeling hungry

Important rules:
- If symptoms suggest urgency (severe pain, chest pain, breathing difficulty), ALWAYS recommend contacting their doctor immediately
- Never suggest changing medication
- Never diagnose conditions
- Keep language simple and encouraging
- Focus on what they CAN do, not what they can't

Provide only the numbered list of recommendations, nothing else.
"""


def get_greeting_prompt(time_of_day: str = "morning") -> str:
    """
    Get a warm greeting prompt based on time of day
    
    Args:
        time_of_day: "morning", "afternoon", or "evening"
        
    Returns:
        Greeting message
    """
    greetings = {
        "morning": "Good morning! I hope you're feeling well today. How are you doing?",
        "afternoon": "Good afternoon! How has your day been so far?",
        "evening": "Good evening! How are you feeling this evening?"
    }
    
    return greetings.get(time_of_day, "Hello! How are you feeling today?")


def get_follow_up_prompt(last_message: str) -> str:
    """
    Generate context-aware follow-up question
    
    Args:
        last_message: User's last message
        
    Returns:
        Relevant follow-up question
    """
    last_lower = last_message.lower()
    
    # Pain mentioned
    if any(word in last_lower for word in ["pain", "hurt", "ache", "sore"]):
        return "On a scale of 1 to 10, how would you rate your pain level?"
    
    # Sleep mentioned
    if any(word in last_lower for word in ["sleep", "tired", "rest"]):
        return "Did you have trouble falling asleep, or did you wake up during the night?"
    
    # Dizziness mentioned
    if any(word in last_lower for word in ["dizzy", "lightheaded", "spinning"]):
        return "Is the dizziness constant or does it come and go?"
    
    # Nausea mentioned
    if any(word in last_lower for word in ["nausea", "nauseous", "sick"]):
        return "Have you been able to eat or drink anything today?"
    
    # Positive response
    if any(word in last_lower for word in ["good", "fine", "well", "okay", "better"]):
        return "That's wonderful to hear! Is there anything else you'd like to share?"
    
    # Default follow-up
    return "I see. Is there anything else that's bothering you today?"


def get_closing_prompt(urgency_level: str = "low") -> str:
    """
    Get appropriate closing message based on urgency
    
    Args:
        urgency_level: "low", "medium", "high", or "urgent"
        
    Returns:
        Closing message
    """
    if urgency_level == "urgent":
        return "Thank you for sharing. Because of what you've told me, I think it's important that you contact your doctor right away. Please call them or have someone help you call. Take care!"
    
    elif urgency_level == "high":
        return "Thank you for sharing. I recommend calling your doctor today to discuss how you're feeling. In the meantime, rest and take it easy. Take care!"
    
    elif urgency_level == "medium":
        return "Thank you for sharing. Please remember to rest and follow the suggestions I gave you. If you don't feel better, don't hesitate to call your doctor. Take care!"
    
    else:  # low
        return "It's great to hear you're doing well! Remember to take your medication as prescribed and reach out anytime you need. Have a wonderful day!"


def get_emergency_prompt() -> str:
    """
    Get emergency escalation message
    
    Returns:
        Emergency message
    """
    return """I'm concerned about the symptoms you've described. This may require immediate medical attention.

Please:
1. Call your doctor right away, OR
2. Have someone call 911 if you're alone and feeling very unwell

If you're experiencing:
- Severe chest pain
- Difficulty breathing
- Sudden severe headache
- Sudden weakness or numbness
- Confusion or trouble speaking

These are emergencies - please get help immediately.

Do you have someone who can help you make the call?"""


def get_medication_reminder_prompt(medication_name: str, time: str) -> str:
    """
    Generate friendly medication reminder
    
    Args:
        medication_name: Name of medication
        time: Scheduled time
        
    Returns:
        Reminder message
    """
    return f"""Hello! This is a friendly reminder that it's time to take your {medication_name}.

The scheduled time is {time}. 

Please take your medication with a glass of water. 

I'll check in with you to see how you're feeling after you take it.

Would you like to take it now?"""
