from api import SimplePipelineAPI

api = SimplePipelineAPI()
result = api.run_simple("Iris", "Class")

print(f"Layout: {result['layout']}")        # e.g., "step_row"
print(f"Accuracy: {result['accuracy']:.2%}")  # e.g., 0.9567 (95.67%)
print(f"Sample image: {result['sample_image_path']}")  # Path to image file