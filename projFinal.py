import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from skimage.io import imread
from skimage.morphology import opening, closing, remove_small_objects, disk
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
import os
import random
import math

# --- CONFIGURAÇÕES GLOBAIS ---
imagens = r"C:\Users\ArturDelpupo\Documents\Faculdade\VisaoComputadores\ProjFinal\Img\Imagens\Img_Base"
sigma_gauss = 1.3
binaryThreshold = 0.63
disk_abrir = 6
disk_fechar = 5
tamanhos_caixas = (2000, 7000, 20000)
proporcoes_caixas = (0.5, 0.8, 1.2, 2.0)
dist_centroide = 103
images = []
index_img = 0
nome_img = "" 
fig = None
botoes = {}
info_caixas = []
unioes_registradas = []


def classificar_proporcao(altura, largura):
    proporcao = altura / largura
    
    if proporcao >= proporcoes_caixas[3] or proporcao <= proporcoes_caixas[0]:
        return "retangular estreita"
    elif (proporcao > proporcoes_caixas[0] and proporcao < proporcoes_caixas[1]) or \
         (proporcao > proporcoes_caixas[2] and proporcao < proporcoes_caixas[3]):
        return "retangular"
    elif proporcao >= proporcoes_caixas[1] and proporcao <= proporcoes_caixas[2]:
        return "quadrado"

def classificar_tamanho(area):
    if tamanhos_caixas[0] <= area < tamanhos_caixas[1]:
        return "pequeno"
    elif tamanhos_caixas[1] <= area < tamanhos_caixas[2]:
        return "médio"
    elif area >= tamanhos_caixas[2]:
        return "grande"

def dist_centroides(caixa1, caixa2):
    cy1 = (caixa1[0] + caixa1[2]) / 2
    cx1 = (caixa1[1] + caixa1[3]) / 2
    cy2 = (caixa2[0] + caixa2[2]) / 2
    cx2 = (caixa2[1] + caixa2[3]) / 2
    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def unir_caixas(caixa1, caixa2):
    linha_min = min(caixa1[0], caixa2[0])
    coluna_min = min(caixa1[1], caixa2[1])
    linha_max = max(caixa1[2], caixa2[2])
    coluna_max = max(caixa1[3], caixa2[3])
    return (linha_min, coluna_min, linha_max, coluna_max)

def unir_caixas_proximas(infos_caixasEncontradas, distancia_maxima):
    global unioes_registradas
    unioes_registradas = []

    if not infos_caixasEncontradas:
        return []

    caixas_atuais = [box['bbox'] for box in infos_caixasEncontradas] 
    
    unido_nesta_iteracao = True
    while unido_nesta_iteracao:
        unido_nesta_iteracao = False
        nova_lista_caixas = []
        caixas_processadas = [False] * len(caixas_atuais)

        for i in range(len(caixas_atuais)):
            if caixas_processadas[i]:
                continue
            
            bbox_i = caixas_atuais[i]
            encontrou_uniao = False
            
            for j in range(i + 1, len(caixas_atuais)):
                if caixas_processadas[j]:
                    continue
                
                bbox_j = caixas_atuais[j]
                dist = dist_centroides(bbox_i, bbox_j)
                
                if dist < distancia_maxima:
                    bbox_unida = unir_caixas(bbox_i, bbox_j)
                    nova_lista_caixas.append(bbox_unida)
                    
                    unioes_registradas.append({
                        'bbox1': bbox_i,
                        'bbox2': bbox_j,
                        'distancia': dist
                    })

                    caixas_processadas[i] = True
                    caixas_processadas[j] = True
                    unido_nesta_iteracao = True
                    encontrou_uniao = True
                    break
            
            if not encontrou_uniao and not caixas_processadas[i]:
                nova_lista_caixas.append(bbox_i)
                caixas_processadas[i] = True
        
        caixas_atuais = nova_lista_caixas
    
    final_boxes_info = []
    for bbox_unida in caixas_atuais:
        min_row, min_col, max_row, max_col = bbox_unida
        width = max_col - min_col
        height = max_row - min_row
        area = width * height
        
        classificacao_proporcao = classificar_proporcao(height, width)
        classificacao_tamanho = classificar_tamanho(area)
        proporcao = height / width

        final_boxes_info.append({
            'bbox': bbox_unida,
            'width_pixels': width,
            'height_pixels': height,
            'area_pixels': area,
            'proporcao': proporcao,
            'classificacao_proporcao': classificacao_proporcao,
            'classificacao_tamanho': classificacao_tamanho
        })
    
    return final_boxes_info

def binarizacao(image):
    image = image[90:, 120:410]

    centro_img = (image.shape[0] // 2, image.shape[1] // 2)
    image_height, image_width = image.shape[:2]

    image_gray = rgb2gray(image)
    image_gauss = gaussian(image_gray, sigma=sigma_gauss)
    image_binarizada = image_gauss > binaryThreshold
    
   
    
    image_diskabrir = opening(image_binarizada, disk(disk_abrir))
    image_diskfechar = closing(image_diskabrir, disk(disk_fechar))
    image_diskfechar = remove_small_objects(image_diskfechar, min_size=tamanhos_caixas[0]//5)

    label_image = label(image_diskfechar)
    regiao = regionprops(label_image)

    caixasPossiveis = []
    for region in regiao: 
        linha_min, coluna_min, linha_max, coluna_max = region.bbox 
        largura = coluna_max - coluna_min
        altura = linha_max - linha_min
        area = region.area

        proporcao = altura / largura
        
        classificacao_proporcao = classificar_proporcao(altura, largura)
        classificacao_tamanho = classificar_tamanho(area)
        
        caixasPossiveis.append({
            'bbox': region.bbox,
            'width_pixels': largura,
            'height_pixels': altura,
            'area_pixels': area,
            'proporcao': proporcao,
            'classificacao_proporcao': classificacao_proporcao,
            'classificacao_tamanho': classificacao_tamanho
        })
    
    caixasUnidas = unir_caixas_proximas(caixasPossiveis, dist_centroide)

    caixasFiltradas = []
    numCaixas_detectadas = len(caixasUnidas)

    if numCaixas_detectadas == 1:
        if caixasUnidas:
            caixasFiltradas.append(caixasUnidas[0])
    elif numCaixas_detectadas == 2:
        for infoCaixas in caixasUnidas:
            linha_min, coluna_min, linha_max, coluna_max = infoCaixas['bbox'] 
            tocarTopo = linha_min <= 2
            tocarEsq = coluna_min <= 2
            tocarDir = coluna_max >= image_width - 2
            
            if not (tocarTopo or tocarEsq or tocarDir):
                caixasFiltradas.append(infoCaixas)
        caixasFiltradas = caixasFiltradas[:2]
    elif numCaixas_detectadas >= 3:
        caixasFiltradas2 = []
        for infoCaixas in caixasUnidas:
            linha_min, coluna_min, linha_max, coluna_max = infoCaixas['bbox'] 
            tocarTopo = linha_min <= 2
            tocarEsq = coluna_min <= 2
            tocarDir = coluna_max >= image_width - 2
            
            if not (tocarTopo or tocarEsq or tocarDir):
                caixasFiltradas2.append(infoCaixas)
        
        caixasFiltradas = caixasFiltradas2[:3]

    global info_caixas
    info_caixas = caixasFiltradas

    return caixasFiltradas, image, image_gray, image_diskfechar, True, binaryThreshold, centro_img, numCaixas_detectadas

def mudar_imagem(event=None):
    global nome_img
    abrir_imagem()

def criar_botoes():
    global botoes, fig
    
    for botao in botoes.values():
        if botao.ax in fig.axes:
            fig.delaxes(botao.ax)
    
    ax_button = fig.add_axes([0.4, 0.05, 0.2, 0.05])
    botoes['next'] = Button(ax_button, 'Próxima Imagem')
    botoes['next'].on_clicked(mudar_imagem)

def abrir_imagem(event=None):
    global fig, index_img, images, nome_img, info_caixas, unioes_registradas

    if not images:
        extensao = ('.png', '.jpg', '.jpeg')
        images = [os.path.join(imagens, f) for f in os.listdir(imagens) 
                    if f.lower().endswith(extensao) and os.path.isfile(os.path.join(imagens, f))]
        random.shuffle(images)

    if index_img >= len(images):
        print("\nAcabaram as imagens!!")
        plt.close(fig)
        return

    image_path = images[index_img]
    nome_img = os.path.basename(image_path)
    index_img += 1
    
    img = imread(image_path)
    info_caixas, cropped_image, grayscale_image, binary_image, success, threshold_used, image_center, num_detected_total = \
        binarizacao(img)

    if fig is None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(bottom=0.20, top=0.9, wspace=0.1, hspace=0.1)
        criar_botoes()
    else:
        for ax in fig.axes:
            if ax not in [btn.ax for btn in botoes.values()]:
                ax.clear()

    eixos = [ax for ax in fig.axes if ax not in [btn.ax for btn in botoes.values()]]
    original, imgray, binaria, caixas = eixos[:4]

    original.imshow(img)
    original.set_title(f"{nome_img}")
    original.axis('off')

    imgray.imshow(grayscale_image, cmap='gray')
    imgray.set_title("Escala de Cinza")
    imgray.axis('off')

    binaria.imshow(binary_image, cmap='gray')
    binaria.set_title(f"Binarizada ({threshold_used:.4f})")
    binaria.axis('off')

    for ind_dist in unioes_registradas:
        caixa1 = ind_dist['bbox1']
        caixa2 = ind_dist['bbox2']
        dist = ind_dist['distancia']

        r1 = mpatches.Rectangle((caixa1[1], caixa1[0]), caixa1[3]-caixa1[1], caixa1[2]-caixa1[0], 
                                fill=False, edgecolor='blue', linewidth=1, linestyle=':')
        r2 = mpatches.Rectangle((caixa2[1], caixa2[0]), caixa2[3]-caixa2[1], caixa2[2]-caixa2[0], 
                                fill=False, edgecolor='blue', linewidth=1, linestyle=':')
        binaria.add_patch(r1)
        binaria.add_patch(r2)

        cy1 = (caixa1[0] + caixa1[2]) / 2
        cx1 = (caixa1[1] + caixa1[3]) / 2
        cy2 = (caixa2[0] + caixa2[2]) / 2
        cx2 = (caixa2[1] + caixa2[3]) / 2

        binaria.plot([cx1, cx2], [cy1, cy2], 'm--', linewidth=1.5, label='Distância União')

        text_x = (cx1 + cx2) / 2
        text_y = (cy1 + cy2) / 2
        binaria.text(text_x, text_y, f'{dist:.2f} px', color='yellow', fontsize=8, 
                     bbox=dict(facecolor='purple', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'), 
                     horizontalalignment='center', verticalalignment='center')

    caixas.imshow(cropped_image)
    caixas.set_title(f"Caixas Detectadas")
    caixas.axis('off')

    for informacao in info_caixas:
        linha_min, coluna_min, linha_max, coluna_max = informacao['bbox'] 
        largura = coluna_max - coluna_min
        altura = linha_max - linha_min
        area = informacao['area_pixels']
        proporcao = informacao['proporcao']
        classificacao_prop = informacao['classificacao_proporcao']
        classificacao_tam = informacao['classificacao_tamanho']

        rect = mpatches.Rectangle((coluna_min, linha_min), largura, altura, fill=False, edgecolor='blue', linewidth=2)
        caixas.add_patch(rect)
        
        caixas.text(coluna_min, linha_min-20, 
                                        f'Área: {area} pixels ({classificacao_tam})\nProporção: {proporcao:.2f} ({classificacao_prop})',
                                        bbox=dict(facecolor='yellow', alpha=0.7), fontsize=9)

    fig.suptitle(f"Caixas Detectadas", fontsize=16,)
    fig.canvas.draw_idle()

    

if __name__ == "__main__":
    abrir_imagem()
    plt.show()