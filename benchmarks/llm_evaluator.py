"""LLM-based evaluation for quality benchmarks.

Uses Claude via AWS Bedrock to actually answer questions about compressed content,
then evaluates if the answers are correct.
"""

import os
import subprocess
from dataclasses import dataclass

import anthropic


@dataclass
class LLMEvaluation:
    """Result of LLM evaluation for a single question."""

    question: str
    expected_answer: str
    llm_answer: str
    is_correct: bool
    confidence: float  # 0-1, how confident the judge is


@dataclass
class LLMEvaluationResult:
    """Aggregated results for all questions on a piece of content."""

    evaluations: list[LLMEvaluation]
    accuracy: float  # % of questions answered correctly
    avg_confidence: float


class LLMEvaluator:
    """Evaluates if compressed content preserves information for LLM tasks."""

    def __init__(self, client: anthropic.Anthropic, model: str):
        """Initialize LLM evaluator.

        Args:
            client: Anthropic client (regular or Bedrock)
            model: Model ID to use
        """
        self.client = client
        self.model = model

    def answer_questions(
        self,
        content: str,
        questions: list[str],
        context_description: str = "code",
    ) -> list[str]:
        """Have the LLM answer questions based on provided content.

        Args:
            content: The compressed content to analyze
            questions: Questions to answer about the content
            context_description: What kind of content this is

        Returns:
            List of answers, one per question
        """
        questions_formatted = "\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(questions)
        )

        prompt = f"""You are analyzing {context_description}. Based ONLY on the content provided below, answer each question concisely.

If the information needed to answer a question is NOT present in the content, respond with "CANNOT_ANSWER" for that question.

<content>
{content}
</content>

<questions>
{questions_formatted}
</questions>

Respond with ONLY the answers, one per line, numbered to match the questions. Keep answers brief (1-2 sentences max)."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse answers from response
        answer_text = response.content[0].text
        lines = [
            line.strip() for line in answer_text.strip().split("\n") if line.strip()
        ]

        # Extract answers, handling numbered format
        answers = []
        for line in lines:
            # Remove leading number and punctuation (e.g., "1. ", "1) ", "1: ")
            if line and line[0].isdigit():
                # Find where the actual answer starts
                for i, char in enumerate(line):
                    if char in ".):- " and i > 0:
                        answer = line[i + 1 :].strip()
                        if answer:
                            answers.append(answer)
                            break
                else:
                    answers.append(line)
            else:
                answers.append(line)

        # Pad with CANNOT_ANSWER if we got fewer answers than questions
        while len(answers) < len(questions):
            answers.append("CANNOT_ANSWER")

        return answers[: len(questions)]

    def judge_answer(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
    ) -> tuple[bool, float]:
        """Judge if an answer is correct.

        Args:
            question: The original question
            expected_answer: What we expect the answer to contain/mean
            actual_answer: What the LLM actually answered

        Returns:
            Tuple of (is_correct, confidence)
        """
        if actual_answer == "CANNOT_ANSWER":
            return False, 1.0

        prompt = f"""You are judging if an answer is correct. Be lenient - the answer doesn't need to be word-for-word, just semantically correct.

Question: {question}

Expected answer (key points that should be covered): {expected_answer}

Actual answer: {actual_answer}

Is the actual answer correct? Consider:
1. Does it address the question?
2. Does it contain the key information from the expected answer?
3. Is it factually consistent with the expected answer?

Respond with ONLY one of these formats:
CORRECT 0.9 (if clearly correct, with confidence 0.7-1.0)
INCORRECT 0.9 (if clearly wrong, with confidence 0.7-1.0)
PARTIAL 0.5 (if partially correct, with confidence 0.3-0.7)"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )

        judgment = response.content[0].text.strip().upper()

        # Parse judgment
        if judgment.startswith("CORRECT"):
            try:
                confidence = float(judgment.split()[1])
            except (IndexError, ValueError):
                confidence = 0.9
            return True, confidence
        elif judgment.startswith("PARTIAL"):
            try:
                confidence = float(judgment.split()[1])
            except (IndexError, ValueError):
                confidence = 0.5
            # Treat partial as correct with lower confidence
            return True, confidence * 0.7
        else:  # INCORRECT or unknown
            try:
                confidence = float(judgment.split()[1])
            except (IndexError, ValueError):
                confidence = 0.9
            return False, confidence

    def evaluate(
        self,
        content: str,
        questions: list[str],
        expected_answers: list[str],
        context_description: str = "code",
    ) -> LLMEvaluationResult:
        """Full evaluation: answer questions and judge correctness.

        Args:
            content: The compressed content to analyze
            questions: Questions to answer
            expected_answers: Expected answers for judging
            context_description: What kind of content this is

        Returns:
            LLMEvaluationResult with all evaluations
        """
        # Get LLM answers
        answers = self.answer_questions(content, questions, context_description)

        # Judge each answer
        evaluations = []
        for question, expected, actual in zip(questions, expected_answers, answers):
            is_correct, confidence = self.judge_answer(question, expected, actual)
            evaluations.append(
                LLMEvaluation(
                    question=question,
                    expected_answer=expected,
                    llm_answer=actual,
                    is_correct=is_correct,
                    confidence=confidence,
                )
            )

        # Calculate aggregates
        correct_count = sum(1 for e in evaluations if e.is_correct)
        accuracy = correct_count / len(evaluations) if evaluations else 0.0
        avg_confidence = (
            sum(e.confidence for e in evaluations) / len(evaluations)
            if evaluations
            else 0.0
        )

        return LLMEvaluationResult(
            evaluations=evaluations,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
        )


def _load_aws_credentials(profile: str) -> dict[str, str]:
    """Load AWS credentials using AWS CLI with credential-process.

    Args:
        profile: AWS profile name to use

    Returns:
        Dictionary with AWS credential environment variables
    """
    try:
        # Use AWS CLI to export credentials in env format
        # This works with credential-process configured profiles (like granted)
        result = subprocess.run(
            ["aws", "configure", "export-credentials", "--format", "env"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "AWS_PROFILE": profile},
        )

        # Parse the exported environment variables from stdout
        # Output format: export AWS_ACCESS_KEY_ID=xxx
        credentials = {}
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]  # Remove "export "
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip("'\"")
                credentials[key] = value

        return credentials

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to get credentials for AWS profile '{profile}': {e.stderr}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError("AWS CLI not found. Please install it.") from e


def create_evaluator(
    use_llm: bool = True,
    aws_profile: str | None = None,
    aws_region: str = "us-west-2",
) -> LLMEvaluator | None:
    """Create an LLM evaluator using AWS Bedrock.

    Args:
        use_llm: Whether to use LLM evaluation
        aws_profile: AWS profile to assume for Bedrock access
        aws_region: AWS region for Bedrock (default: us-west-2)

    Returns:
        LLMEvaluator instance or None if disabled/unavailable
    """
    if not use_llm:
        return None

    if not aws_profile:
        print("Warning: --aws-profile not set, LLM evaluation disabled")
        return None

    try:
        # Load AWS credentials using assume CLI
        print(f"Assuming AWS profile '{aws_profile}'...")
        credentials = _load_aws_credentials(aws_profile)

        access_key = credentials.get("AWS_ACCESS_KEY_ID", "")
        secret_key = credentials.get("AWS_SECRET_ACCESS_KEY", "")
        session_token = credentials.get("AWS_SESSION_TOKEN")

        # Create Bedrock client with explicit credentials
        client = anthropic.AnthropicBedrock(
            aws_access_key=access_key,
            aws_secret_key=secret_key,
            aws_session_token=session_token,
            aws_region=aws_region,
        )

        # Use Claude 3 Haiku on Bedrock (usually enabled by default)
        model = "us.anthropic.claude-3-haiku-20240307-v1:0"

        print(f"Using Bedrock model: {model} in {aws_region}")
        return LLMEvaluator(client=client, model=model)

    except Exception as e:
        print(f"Warning: Failed to create Bedrock evaluator: {e}")
        return None
