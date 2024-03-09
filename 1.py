from generator.simplified_model import SimpleRetrievalAugmentedGenerator
from transformers.generation.utils import GenerateEncoderDecoderOutput

model_path = "/data/workspace/muning/GloveInDark/models/leandojo-lean4-tacgen-byt5-small"

generator = SimpleRetrievalAugmentedGenerator(model_path, 1, 1024, 1024)

generator.generate("a+b", None, None, None, 1)