import torch
from torchvision.models import resnet18

model = resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')


#if __name__ == '__main__':
    #do some tests
    #input = torch.random(3, 224, 224)
    #output = model(input)
    #script_output = script_model(input)
    #__, unscripted_top5_indices = torch.topk(output, 5)
    #__, scripted_top5_indices = torch.topk(script_output, 5)
    #assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)
