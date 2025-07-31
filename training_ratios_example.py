"""
Example usage of the new training ratios functionality in fusGAN

This script demonstrates how to use the modified GAN class with different
training ratios for the discriminator and generator.
"""

import tensorflow as tf
from models import GAN, Generator, Discriminator

def create_example_gan():
    """Create a simple GAN for demonstration"""
    # Example parameters - adjust according to your needs
    input_channels = 3
    output_channels = 1
    
    # Create models
    generator = Generator(input_channels, output_channels)
    discriminator = Discriminator(input_channels)
    
    # Create optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # Create loss function
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Create GAN
    gan = GAN(
        generator=generator.model,
        discriminator=discriminator.model,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        loss_object=loss_object
    )
    
    return gan

def training_examples():
    """Demonstrate different training patterns"""
    
    # Create a dummy dataset for demonstration
    def create_dummy_dataset():
        # This is just for demonstration - replace with your actual dataset
        def generator():
            for _ in range(100):  # 100 batches
                input_img = tf.random.normal([8, 128, 128, 3])  # batch_size=8
                target_img = tf.random.normal([8, 128, 128, 1])
                yield input_img, target_img
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=[8, 128, 128, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[8, 128, 128, 1], dtype=tf.float32)
            )
        )
        return dataset
    
    train_dataset = create_dummy_dataset()
    gan = create_example_gan()
    
    print("Training Examples:")
    print("=" * 50)
    
    # Example 1: Standard 1:1 training (same as before)
    print("1. Standard 1:1 training (discriminator:generator = 1:1)")
    # gan.fit(train_dataset, epochs=1)
    
    # Example 2: Train discriminator more frequently (2:1 ratio)
    print("2. Train discriminator 2x more (discriminator:generator = 2:1)")
    # gan.fit(train_dataset, epochs=1, discriminator_ratio=2, generator_ratio=1)
    
    # Example 3: Train generator more frequently (1:2 ratio)
    print("3. Train generator 2x more (discriminator:generator = 1:2)")
    # gan.fit(train_dataset, epochs=1, discriminator_ratio=1, generator_ratio=2)
    
    # Example 4: Heavy discriminator training (5:1 ratio)
    print("4. Heavy discriminator training (discriminator:generator = 5:1)")
    # gan.fit(train_dataset, epochs=1, discriminator_ratio=5, generator_ratio=1)
    
    # Example 5: Alternating block training
    print("5. Alternating block training (3 disc steps, then 2 gen steps)")
    # gan.fit_with_alternating_training(train_dataset, epochs=1, disc_steps=3, gen_steps=2)
    
    # Example 6: Individual training steps
    print("6. Manual control over individual training steps")
    for i, (input_image, target) in enumerate(train_dataset.take(10)):
        if i % 3 == 0:  # Train discriminator every 3rd step
            gen_loss, disc_loss, l1_loss, real_acc, gen_acc = gan.train_step(
                input_image, target, train_discriminator=True, train_generator=False)
            print(f"Step {i}: Trained discriminator only - Disc Loss: {disc_loss:.4f}")
        else:  # Train generator otherwise
            gen_loss, disc_loss, l1_loss, real_acc, gen_acc = gan.train_step(
                input_image, target, train_discriminator=False, train_generator=True)
            print(f"Step {i}: Trained generator only - Gen Loss: {gen_loss:.4f}")

def usage_guide():
    """Print usage guide for the new functionality"""
    print("\nUsage Guide:")
    print("=" * 50)
    print("""
    The GAN class now supports flexible training ratios:
    
    1. fit() method with ratios:
       gan.fit(dataset, epochs=10, discriminator_ratio=2, generator_ratio=1)
       - This trains discriminator 2 times for every 1 generator training
       - Ratios determine the training pattern within each cycle
    
    2. fit_with_alternating_training() method:
       gan.fit_with_alternating_training(dataset, epochs=10, disc_steps=3, gen_steps=2)
       - This trains discriminator for 3 consecutive steps, then generator for 2 steps
       - Creates block-based alternating pattern
    
    3. Individual training control:
       gan.train_step(input, target, train_discriminator=True, train_generator=False)
       gan.train_step(input, target, train_discriminator=False, train_generator=True)
       - Full manual control over what gets trained
    
    4. New separate training methods:
       gan.discriminator_train_step(input, target)  # Train only discriminator
       gan.generator_train_step(input, target)      # Train only generator
    
    Common training patterns:
    - discriminator_ratio=2, generator_ratio=1: Good when discriminator is too weak
    - discriminator_ratio=1, generator_ratio=2: Good when discriminator is too strong
    - discriminator_ratio=5, generator_ratio=1: Heavy discriminator training
    - alternating blocks: Good for preventing mode collapse
    """)

if __name__ == "__main__":
    print("fusGAN Training Ratios Example")
    print("=" * 50)
    
    usage_guide()
    training_examples()
    
    print("\nAll examples completed successfully!")
