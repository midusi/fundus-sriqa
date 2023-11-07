from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from utils.plot import add_colorbar



def torch2numpy_image(image,clip=False):
  image = image.detach().cpu().numpy()
  if clip:
    image = np.clip(image,0,1)
  image = np.transpose(image, (1, 2, 0))
  return image


def torch2numpy_data(image,grad,gradcam):
  return torch2numpy_image(image,clip=True),torch2numpy_image(grad),torch2numpy_image(gradcam)
    

def get_important_activations(activations,k):

  activation_importance = np.abs(activations).sum(axis=(0,1))
  
  activation_importance_percentage = activation_importance/activation_importance.sum()
  indices=np.argsort(activation_importance)
  indices = np.flip(indices[-k:])
  return activations[:,:,indices],activation_importance[indices],activation_importance_percentage[indices],indices
 
def plot_activations(filepath,image,label,activations,id,rowscols=5):
  image = torch2numpy_image(image,clip=True)
  activations = torch2numpy_image(activations)
  rowscols = min(rowscols,int(np.sqrt(activations.shape[2])))
  dpi_factor=4
  f,activations_ax = plt.subplots(rowscols,rowscols,figsize=(rowscols*dpi_factor,rowscols*dpi_factor))
  
  activations_ax = [ax  for rowax in activations_ax for ax in rowax]
  k = len(activations_ax)
  
  np.save(f"{label}_{id}_activations.npz",activations)

  important_activations,importance,relative_importance,indices = get_important_activations(activations,k)
  for i,ax in enumerate(activations_ax):
    feature_map = important_activations[:,:,i]
    # print(feature_map.shape,image.shape)
    feature_map = resize(feature_map, image.shape[:2])
    feature_map = feature_map[:,:,np.newaxis]
    # print(feature_map.shape)
    title = f"Act {indices[i]}, \n |mag|={importance[i]:2f},\n |rmag|={relative_importance[i]*100:.2f}%"
    viz.visualize_image_attr(feature_map,image,method="blended_heat_map", sign="absolute_value",
                          show_colorbar=True, title=title,plt_fig_axis=(f,ax),use_pyplot=False)
    
    # ax.imshow(important_activations[:,:,i])
    # ax.set_title(title,fontsize=5)

  # plt.suptitle(f"Filtros con mayor magnitud, {relative_importance.sum()*100:.2f}% de la magnitud total",fontsize=5*dpi_factor)
  plt.tight_layout()
  plt.savefig(filepath)
  plt.close()
  


def plot_saliency_paper(filepath,image,methods,method_names,class_name,width=5,title_fontsize=30):
    
    n = len(methods)
    f,axis = plt.subplots(1,n+1,dpi=250,figsize=((n+1)*width,width),squeeze=False)
    axis=axis[0,:]
    plt.xticks([])
    plt.yticks([])
    ax_image,ax_methods = axis[0],axis[1:]

    ax_image.imshow(image)
    ax_image.set_axis_off()

    ax_image.set_title("Original",fontsize=title_fontsize)
    

    for ax,method,name in zip(ax_methods,methods,method_names):
      if name=="GradCAM":
        sign = "all"
        alpha=0.5
      else:
        sign = "all"
        alpha =0.5
      
      if name=="RISE":
        sign = "positive"
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
                "WhGn", ["white", "green"]
            )#"Greens"#"plasma"
      else:
        cmap = None

      viz.visualize_image_attr(method, image, method="blended_heat_map", sign=sign,
                            show_colorbar=True, title=name,plt_fig_axis=(f,ax),use_pyplot=False,alpha_overlay=alpha,cmap=cmap)
      cbar = ax.images[-1].colorbar
      cbar.ax.tick_params(labelsize=20) 
      ax.set_title(name,fontsize= title_fontsize)                                
      ax.set_axis_off()
      # title = ax.get_title()                        
      # ax.set_title(title,fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def  plot_average_saliency(image,saliency,gradcam,id,filepath,model_name,class_name):

    image,saliency,gradcam=torch2numpy_image(image,clip=True),torch2numpy_image(saliency),torch2numpy_image(gradcam)
    f,(a1,a2,a3)=plt.subplots(1,3,figsize=(15,5))
    a1.imshow(image)
    a1.set_title("Average input")
    viz.visualize_image_attr(saliency,method="heat_map", sign="all",
                          show_colorbar=True, title="Gradient",plt_fig_axis=(f,a2),use_pyplot=False
                          ,alpha_overlay=1) 
    viz.visualize_image_attr(gradcam,method="heat_map", sign="all",
    show_colorbar=True, title="GradCAM",plt_fig_axis=(f,a3),use_pyplot=False
    ,alpha_overlay=1) 

    for ax in [a1,a2,a3]:
      # title = ax.get_title()                        
      # ax.set_title(title,fontsize=8)
      ax.set_axis_off()
    # a1.imshow(average_saliency)
    # a1.set_title(f"Gradient")
    # a2.imshow(average_gradcam)
    # a2.set_title(f"Gradcam")
    # plt.suptitle(f"Valores promedio para la clase {class_name} y el modelo {model_name}")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()