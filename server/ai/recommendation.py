// ...existing imports...

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
               