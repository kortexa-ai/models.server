from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("nano_v3")
class NanoV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and (
                request.chat_template_kwargs.get("enable_thinking") is False
                or request.chat_template_kwargs.get("force_nonempty_content") is True
            )
            and final_content is None
        ):
            # When thinking is disabled or force_nonempty_content is set,
            # place all content in the response body rather than reasoning
            reasoning_content, final_content = None, reasoning_content

        return reasoning_content, final_content
