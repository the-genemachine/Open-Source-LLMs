# Vision Adapters

A **vision adapter** is an essential component that enables language models to process and interpret image inputs, effectively transforming them into multimodal models capable of handling both text and visual data. In LM Studio, integrating a vision adapter allows you to interact with models that can analyze and describe images, enhancing the versatility of your AI applications.

**Using a Vision Adapter in LM Studio:**

1. **Select a Vision-Enabled Model:**
   - In LM Studio, navigate to the "Discover" tab (represented by a magnifying glass icon) to access the model search feature.
   - Look for models marked with a small yellow eye icon, indicating they are vision-enabled and support image input.
   - For example, models like "Llava v1.5 7B" and "Phi 3.5 Vision Instruct" are known to support image inputs.

2. **Download the Model and Vision Adapter:**
   - When downloading a vision-enabled model, ensure that both the primary model file and its corresponding vision adapter file are downloaded.
   - The vision adapter file often has a prefix like "mmproj-" (MultiModal Projector) in its name.
   - In some cases, you may need to download these files manually and place them in the LM Studio model directory.

3. **Load the Model in LM Studio:**
   - After downloading, load the model into LM Studio by selecting it from the model list.
   - Ensure that the vision adapter is correctly associated with the model to enable image processing capabilities.

4. **Interact with the Model Using Images:**
   - With the vision-enabled model loaded, an image import button should appear at the bottom of the prompt input window in LM Studio.
   - Click this button to upload an image from your computer.
   - You can send the image alone or accompany it with a text prompt for more specific interactions.

**Considerations:**

- **Model Compatibility:** Not all models support vision adapters. Ensure you select models specifically designed for image input, as indicated by the yellow eye icon or the presence of a vision adapter file.

- **Performance Variations:** The accuracy of image interpretation can vary depending on the model's size and training. Smaller models may struggle with complex images or large amounts of text within images. Experiment with different models to find one that meets your requirements.
