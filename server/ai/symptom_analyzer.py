import json
import logging
from server.models.schemas import SymptomExtractionResult
from server.ai.llm_client import llm_client
from server.ai.prompts import get_symptom_extraction_prompt

logger = logging.getLogger(__name__)

# Try to import spaCy, but don't fail if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available - using fallback entity extraction")

class SymptomAnalyzer:
    def __init__(self):
        # Load spaCy model for entity extraction if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for entity extraction")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}. Using fallback.")
                self.nlp = None
        
        # Define symptom keywords
        self.symptom_keywords = {
            "pain": ["pain", "ache", "hurt", "sore", "painful", "aching"],
            "nausea": ["nausea", "nauseous", "queasy", "sick to stomach"],
            "dizziness": ["dizzy", "dizziness", "lightheaded", "vertigo", "spinning"],
            "fatigue": ["tired", "fatigue", "exhausted", "weak", "weakness", "fatigued"],
            "headache": ["headache", "head pain", "migraine"],
            "shortness_of_breath": ["breathless", "short of breath", "breathing", "breath"],
            "confusion": ["confused", "confusion", "disoriented", "forgetful"],
            "chest_pain": ["chest pain", "chest pressure", "tight chest"]
        }

    async def extract_symptoms(
        self,
        conversation_text: str
    ) -> SymptomExtractionResult:
        """
        Extract symptoms from conversation using NLP and LLM
        """
        # Use NLP for entity extraction if available
        entities = []
        if self.nlp:
            try:
                doc = self.nlp(conversation_text)
                entities = [ent.text for ent in doc.ents]
            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {e}")
        
        # Extract keywords
        detected_symptoms = self._extract_keywords(conversation_text.lower())

        # Use LLM for structured extraction
        extraction_prompt = get_symptom_extraction_prompt(conversation_text)
        
        try:
            llm_response = await llm_client.generate_response([
                {"role": "system", "content": "You are a medical data extraction assistant. Always respond with valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ], temperature=0.3)  # Lower temperature for more consistent JSON

            # Parse LLM response
            # Remove markdown code blocks if present
            llm_response = llm_response.strip()
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:]
            if llm_response.startswith("```"):
                llm_response = llm_response[3:]
            if llm_response.endswith("```"):
                llm_response = llm_response[:-3]
            
            symptoms_data = json.loads(llm_response.strip())
        except Exception as e:
            logger.error(f"Error parsing LLM symptom extraction: {e}")
            # Fallback to keyword-based extraction
            symptoms_data = {
                "symptoms": detected_symptoms,
                "vital_signs": {},
                "concerns": [],
                "urgency_level": "low",
                "notes": "Keyword-based extraction fallback"
            }

        # Merge keyword detection with LLM extraction
        final_symptoms = {**detected_symptoms, **symptoms_data.get("symptoms", {})}

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(
            final_symptoms,
            detected_symptoms,
            symptoms_data.get("symptoms", {})
        )

        # Calculate urgency
        urgency = symptoms_data.get("urgency_level", self._calculate_urgency(final_symptoms))

        # Analyze sentiment
        sentiment = self._analyze_sentiment(conversation_text)

        return SymptomExtractionResult(
            symptoms=final_symptoms,
            confidence_scores=confidence_scores,
            extracted_entities=entities,
            sentiment=sentiment,
            urgency_level=urgency
        )

    def _extract_keywords(self, text: str) -> dict:
        """Extract symptoms using keyword matching"""
        detected = {}
        
        for symptom, keywords in self.symptom_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected[symptom] = True
        
        # Extract pain level if mentioned
        import re
        pain_pattern = r'(\d+)\s*(?:out of|/)\s*10'
        match = re.search(pain_pattern, text)
        if match:
            detected["pain_level"] = int(match.group(1))
        
        return detected

    def _calculate_confidence(
        self,
        final_symptoms: dict,
        keyword_symptoms: dict,
        llm_symptoms: dict
    ) -> dict:
        """Calculate confidence scores for extracted symptoms"""
        confidence = {}
        
        for symptom, value in final_symptoms.items():
            # Higher confidence if both methods detected it
            if symptom in keyword_symptoms and symptom in llm_symptoms:
                confidence[symptom] = 0.9
            # Medium confidence if only one method detected it
            elif symptom in keyword_symptoms or symptom in llm_symptoms:
                confidence[symptom] = 0.6
            # Lower confidence if inferred
            else:
                confidence[symptom] = 0.3
        
        return confidence

    def _calculate_urgency(self, symptoms: dict) -> str:
        """Calculate urgency level from symptoms"""
        # Check for urgent symptoms
        urgent_symptoms = [
            "chest_pain", "difficulty_breathing", "severe_pain",
            "confusion", "sudden_weakness", "shortness_of_breath"
        ]
        
        high_symptoms = [
            "severe_headache", "high_fever", "persistent_vomiting"
        ]

        # Check pain level
        pain_level = symptoms.get("pain_level", 0)
        if pain_level >= 8:
            return "urgent"
        elif pain_level >= 6:
            return "high"

        # Check for urgent symptoms
        if any(symptoms.get(s) for s in urgent_symptoms):
            return "urgent"
        
        # Check for high priority symptoms
        if any(symptoms.get(s) for s in high_symptoms):
            return "high"
        
        # Count total symptoms
        symptom_count = sum(1 for v in symptoms.values() if v is True)
        
        if symptom_count >= 4:
            return "high"
        elif symptom_count >= 2:
            return "medium"
        else:
            return "low"

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of conversation"""
        # Simple keyword-based sentiment (could use transformers for better results)
        positive_words = ["good", "better", "fine", "well", "great", "improved", "happy"]
        negative_words = ["bad", "worse", "pain", "dizzy", "nausea", "tired", "awful", "terrible"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    async def track_symptom_trends(
        self,
        user_id: str,
        days: int = 7
    ) -> dict:
        """Track symptom trends over time"""
        # Load recent health logs for user
        # This would integrate with health_logs.json
        # For now, return placeholder structure
        
        return {
            "user_id": user_id,
            "period_days": days,
            "trending_up": [],  # Symptoms getting worse
            "trending_down": [],  # Symptoms improving
            "new_symptoms": [],  # Recently appeared
            "resolved_symptoms": [],  # No longer reported
            "persistent_symptoms": []  # Consistently reported
        }

symptom_analyzer = SymptomAnalyzer()
