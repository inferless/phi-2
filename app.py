from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):
        model_id = "Inferless/inferless-phi-2-DPO"  # Specify the model repository ID
        # Define sampling parameters for model generation
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
        # Initialize the LLM object
        self.llm = LLM(model=model_id)
        
    def infer(self,inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        result = self.llm.generate(prompts, self.sampling_params)
        # Extract the generated text from the result
        generated_outputs = [output.outputs[0].text for output in results]

        if len(prompts) == 1:
            return {'generated_result': generated_outputs[0]}
        return {'generated_results': generated_outputs}

    def finalize(self):
        pass
