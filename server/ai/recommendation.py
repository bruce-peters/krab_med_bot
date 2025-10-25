from typing import List
import logging
from server.models.schemas import HealthRecommendation
from server.ai.llm_client import llm_client
from server.ai.prompts import get_recommendation_prompt

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        # Define recommendation templates
        self.templates = {
            "hydration_low": "Drink a glass of water now. Staying hydrated helps your body process medications better.",
            "hydration_medium": "Try to drink 6-8 glasses of water throughout the day. Keep a water bottle nearby.",
            "rest_fatigue": "Your body needs rest to heal. Try to take a short nap if you're feeling tired.",
            "rest_poor_sleep": "Consider going to bed 30 minutes earlier tonight. A consistent sleep schedule helps.",
            "activity_gentle": "A short 5-minute walk around your home can help with circulation and mood.",
            "seek_help_urgent": "Please call your doctor right away. These symptoms need medical attention.",
            "seek_help_high": "Consider calling your doctor today to discuss these symptoms.",
            "nutrition_nausea": "Try eating small, bland meals like crackers or toast. Avoid heavy or spicy foods.",
            "nutrition_appetite": "Even if you're not hungry, try to eat small amounts every few hours."
        }

    async def generate_recommendations(
        self,
        symptoms: dict,
        user_history: dict
    ) -> List[HealthRecommendation]:
        """
        Generate personalized health recommendations
        """
        # Build context
        prompt = get_recommendation_prompt(symptoms, user_history)

        # Get LLM recommendations
        try:
            llm_response = await llm_client.generate_response([
                {"role": "system", "content": "You are a helpful health advisor for elderly patients. Provide simple, actionable recommendations."},
                {"role": "user", "content": prompt}
            ], temperature=0.7, max_tokens=400)
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Fallback to template-based recommendations
            return self._generate_template_recommendations(symptoms)

        # Parse and structure recommendations
        recommendations = self._parse_recommendations(llm_response, symptoms)
        
        # Add urgent medical attention if needed
        if self._needs_urgent_care(symptoms):
            urgent_rec = HealthRecommendation(
                recommendation_text="Please contact your doctor or healthcare provider immediately to discuss these symptoms.",
                category="seek_help",
                priority="high",
                based_on=list(symptoms.keys())
            )
            recommendations.insert(0, urgent_rec)

        return recommendations

    def _parse_recommendations(
        self,
        llm_text: str,
        symptoms: dict
    ) -> List[HealthRecommendation]:
        """Parse LLM text into structured recommendations"""
        # Split by bullet points or numbers
        lines = [line.strip() for line in llm_text.split('\n') if line.strip()]

        recommendations = []
        for line in lines:
            # Skip headers or empty lines
            if len(line) < 10 or line.endswith(':'):
                continue
                
            if line.startswith(('-', '•')) or (line[0].isdigit() and line[1] in '.)'):
                # Clean the line
                clean_line = line.lstrip('-•0123456789. ').strip()

                # Skip if too short
                if len(clean_line) < 10:
                    continue

                # Categorize
                category = self._categorize_recommendation(clean_line)
                priority = self._assess_priority(clean_line, symptoms)

                rec = HealthRecommendation(
                    recommendation_text=clean_line,
                    category=category,
                    priority=priority,
                    based_on=list(symptoms.keys())
                )
                recommendations.append(rec)

        return recommendations[:5]  # Limit to top 5 recommendations

    def _generate_template_recommendations(self, symptoms: dict) -> List[HealthRecommendation]:
        """Generate recommendations using templates as fallback"""
        recommendations = []
        
        # Pain recommendations
        if symptoms.get("pain_level", 0) > 3:
            rec = HealthRecommendation(
                recommendation_text="Rest in a comfortable position. Apply a warm compress if appropriate for your type of pain.",
                category="rest",
                priority="medium",
                based_on=["pain_level"]
            )
            recommendations.append(rec)
        
        # Fatigue recommendations
        if symptoms.get("fatigue"):
            rec = HealthRecommendation(
                recommendation_text=self.templates["rest_fatigue"],
                category="rest",
                priority="medium",
                based_on=["fatigue"]
            )
            recommendations.append(rec)
        
        # Hydration for various symptoms
        if any(symptoms.get(s) for s in ["nausea", "dizziness", "fatigue"]):
            rec = HealthRecommendation(
                recommendation_text=self.templates["hydration_medium"],
                category="hydration",
                priority="medium",
                based_on=["nausea", "dizziness"]
            )
            recommendations.append(rec)
        
        return recommendations

    def _categorize_recommendation(self, text: str) -> str:
        """Categorize recommendation"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["water", "drink", "hydrate", "fluid"]):
            return "hydration"
        elif any(word in text_lower for word in ["rest", "sleep", "relax", "lie down", "nap"]):
            return "rest"
        elif any(word in text_lower for word in ["walk", "exercise", "move", "stretch", "activity"]):
            return "activity"
        elif any(word in text_lower for word in ["doctor", "physician", "medical", "call", "contact", "emergency"]):
            return "seek_help"
        elif any(word in text_lower for word in ["eat", "food", "meal", "nutrition"]):
            return "nutrition"
        else:
            return "general"

    def _assess_priority(self, text: str, symptoms: dict) -> str:
        """Assess recommendation priority"""
        text_lower = text.lower()
        
        # High priority keywords
        if any(word in text_lower for word in ["immediately", "urgent", "emergency", "right away", "call doctor"]):
            return "high"
        
        # Check symptom severity
        if symptoms.get("pain_level", 0) >= 7:
            return "high"
        
        # Medium priority
        if any(word in text_lower for word in ["today", "soon", "important", "should"]):
            return "medium"
        
        # Default to low
        return "low"

    def _needs_urgent_care(self, symptoms: dict) -> bool:
        """Determine if symptoms require urgent medical attention"""
        urgent_symptoms = [
            "chest_pain",
            "difficulty_breathing",
            "shortness_of_breath",
            "confusion",
            "sudden_weakness"
        ]
        
        # Check for urgent symptoms
        if any(symptoms.get(s) for s in urgent_symptoms):
            return True
        
        # Check pain level
        if symptoms.get("pain_level", 0) >= 8:
            return True
        
        return False

    async def prioritize_recommendations(
        self,
        recommendations: List[HealthRecommendation]
    ) -> List[HealthRecommendation]:
        """Sort recommendations by priority"""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        
        return sorted(
            recommendations,
            key=lambda r: priority_order.get(r.priority, 3)
        )

# Create singleton instance
recommendation_engine = RecommendationEngine()
