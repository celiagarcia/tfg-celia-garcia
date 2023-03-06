# Detección de Células mediante Deep Learning en Secuencias de Vídeo-Microscopía

Trabajo Fin de Grado

Grado en Ingeniería en Sistemas Audiovisuales y Multimedia, ETSIT, URJC.

## Descripción 

Técnica de segmentación de células en imágenes de vídeo microscopía, utilizando Deep Learning a partir de una arquitectura de Red Neuronal Convolucional (UNet), utilizando el repositorio de datos que proporciona http://celltrackingchallenge.net/.

## Dataset

Dataset para el entrenamiento. Imágenes y etiquetas:

https://user-images.githubusercontent.com/9032799/178141268-4a7d30b5-99e3-4ccc-8d1d-4db28871e53c.mp4

Secuencias de imágenes no etiquetadas para evaluar resultados:

https://user-images.githubusercontent.com/9032799/178141362-f844dcc1-7a2b-452d-9817-98991d638163.mp4

## Resultados

Se ha entrenado el modelo con aumento de datos y sin aumento de datos, además con diferentes números de epochs. Comparativa de resultados:

https://user-images.githubusercontent.com/9032799/178141448-41b71d82-775f-49bd-86f5-719df9935ae5.mp4

https://user-images.githubusercontent.com/9032799/178141452-8bd783ac-f836-4b74-b9cd-760e7562c839.mp4

## Referencias

https://academic.oup.com/bioinformatics/article/30/11/1609/283435

https://www.nature.com/articles/nmeth.4473
