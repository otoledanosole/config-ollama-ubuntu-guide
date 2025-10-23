# Guía Completa: Configuración de Ollama en Ubuntu 24.04 para Hardware de Alto Rendimiento

**Versión:** 1.0  
**Fecha:** Octubre 2025  
**Autor:** Oscar Toledano Sole   
**Hardware objetivo:** AMD Ryzen 9 3900X, 128GB RAM, NVIDIA RTX 5060 Ti 16GB VRAM  
**Sistema Operativo:** Ubuntu 24.04 LTS

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
   - 1.1 [Objetivos de esta guía](#11-objetivos-de-esta-guía)
   - 1.2 [Conceptos fundamentales de LLMs](#12-conceptos-fundamentales-de-llms)
   - 1.3 [Hardware y especificaciones](#13-hardware-y-especificaciones)
2. [Preparación del Sistema](#2-preparación-del-sistema)
   - 2.1 [Actualización del sistema](#21-actualización-del-sistema)
   - 2.2 [Verificación de hardware](#22-verificación-de-hardware)
   - 2.3 [Herramientas esenciales](#23-herramientas-esenciales)
3. [Instalación de Drivers NVIDIA](#3-instalación-de-drivers-nvidia)
   - 3.1 [Identificación de GPU](#31-identificación-de-gpu)
   - 3.2 [Limpieza de instalaciones previas](#32-limpieza-de-instalaciones-previas)
   - 3.3 [Instalación de drivers](#33-instalación-de-drivers)
   - 3.4 [Verificación y testing](#34-verificación-y-testing)
4. [Instalación de CUDA Toolkit](#4-instalación-de-cuda-toolkit)
   - 4.1 [Instalación de CUDA 12.x](#41-instalación-de-cuda-12x)
   - 4.2 [Configuración de variables de entorno](#42-configuración-de-variables-de-entorno)
   - 4.3 [Instalación de cuDNN](#43-instalación-de-cudnn)
5. [Instalación de Ollama](#5-instalación-de-ollama)
   - 5.1 [Instalación básica](#51-instalación-básica)
   - 5.2 [Configuración del servicio](#52-configuración-del-servicio)
   - 5.3 [Verificación de GPU](#53-verificación-de-gpu)
6. [Configuración Avanzada de Ollama](#6-configuración-avanzada-de-ollama)
   - 6.1 [Variables de entorno del servicio](#61-variables-de-entorno-del-servicio)
   - 6.2 [Configuración de usuario](#62-configuración-de-usuario)
   - 6.3 [Optimización de memoria](#63-optimización-de-memoria)
7. [Selección y Descarga de Modelos](#7-selección-y-descarga-de-modelos)
   - 7.1 [Modelos recomendados para RAG](#71-modelos-recomendados-para-rag)
   - 7.2 [Modelos para conversación](#72-modelos-para-conversación)
   - 7.3 [Script de descarga automatizada](#73-script-de-descarga-automatizada)
   - 7.4 [Gestión de modelos](#74-gestión-de-modelos)
8. [Optimizaciones del Sistema](#8-optimizaciones-del-sistema)
   - 8.1 [Configuración de memoria](#81-configuración-de-memoria)
   - 8.2 [Governor de CPU](#82-governor-de-cpu)
   - 8.3 [Configuración de GPU](#83-configuración-de-gpu)
   - 8.4 [Herramientas de monitoreo](#84-herramientas-de-monitoreo)
9. [Testing y Benchmarking](#9-testing-y-benchmarking)
   - 9.1 [Tests básicos](#91-tests-básicos)
   - 9.2 [Benchmark de modelos](#92-benchmark-de-modelos)
   - 9.3 [Test de RAG completo](#93-test-de-rag-completo)
10. [Troubleshooting](#10-troubleshooting)
    - 10.1 [Problemas comunes](#101-problemas-comunes)
    - 10.2 [Comandos de diagnóstico](#102-comandos-de-diagnóstico)
    - 10.3 [Reset completo](#103-reset-completo)
11. [Scripts de Utilidad](#11-scripts-de-utilidad)
12. [Configuración de Red y API](#12-configuración-de-red-y-api)
13. [Referencias y Recursos](#13-referencias-y-recursos)
14. [Apéndices](#14-apéndices)

---

## 1. Introducción

### 1.1 Objetivos de esta guía

Esta guía proporciona instrucciones detalladas para configurar un entorno optimizado de ejecución de Large Language Models (LLMs) utilizando Ollama en Ubuntu 24.04. El objetivo es maximizar el rendimiento del hardware disponible para aplicaciones de:

- **Retrieval-Augmented Generation (RAG)**: Sistemas que combinan búsqueda de información con generación de texto
- **Conversación avanzada**: Asistentes conversacionales con memoria y contexto
- **Análisis de datos**: Procesamiento y análisis de información estructurada y no estructurada

### 1.2 Conceptos fundamentales de LLMs

#### ¿Qué son los parámetros?

Los parámetros son los pesos de la red neuronal que el modelo aprendió durante el entrenamiento. Un modelo con 12 billones (12B) de parámetros tiene 12,000 millones de números que determinan cómo procesa la información.

**Impacto de los parámetros:**
- **Más parámetros → Mayor capacidad**: Mejor comprensión del lenguaje y razonamiento
- **Más parámetros → Más recursos**: Mayor uso de memoria RAM/VRAM
- **Más parámetros → Menor velocidad**: Más cálculos por token generado

#### Tamaño del modelo en disco

El tamaño en disco depende de:
1. **Número de parámetros**: Base del tamaño
2. **Precisión numérica**: Bits usados por parámetro
3. **Cuantización aplicada**: Reducción de precisión

**Ejemplo con Gemma 3:12B:**
- Sin cuantizar (FP16): ~24GB (16 bits × 12B parámetros)
- Cuantizado Q8: ~8GB (8 bits × 12B parámetros ≈ 50% reducción)
- Cuantizado Q4: ~4GB (4 bits × 12B parámetros ≈ 25% del original)

#### Ventana de contexto

Es la cantidad de tokens (fragmentos de texto) que el modelo puede "recordar" en una conversación.

**Conversiones aproximadas:**
- 1 token ≈ 0.75 palabras
- 128k tokens ≈ 96,000 palabras ≈ 200-300 páginas

**Impacto:**
- **Contexto mayor → Mayor memoria**: Crece cuadráticamente con el tamaño
- **Contexto mayor → Mejor comprensión**: Puede procesar documentos más largos
- **Contexto mayor → Más lento**: Especialmente con contextos muy llenos

#### Cuantización

La cuantización reduce la precisión numérica de los parámetros para ahorrar memoria con mínima pérdida de calidad.

**Niveles de cuantización:**
- **FP16/FP32**: Sin cuantizar, máxima calidad
- **Q8_0**: 8 bits, ~50% tamaño, pérdida mínima (~1-2%)
- **Q5_K_M**: 5 bits, ~30% tamaño, pérdida baja (~3-5%)
- **Q4_K_M**: 4 bits, ~25% tamaño, pérdida moderada (~5-10%)
- **Q4_0**: 4 bits simple, ~25% tamaño, mayor pérdida
- **Q3/Q2**: No recomendado, pérdida significativa de calidad

**Recomendación:** Para tu hardware, Q4_K_M ofrece el mejor balance entre calidad y eficiencia.

### 1.3 Hardware y especificaciones

#### Especificaciones del sistema objetivo

| Componente | Especificación | Impacto en LLMs |
|------------|----------------|-----------------|
| **CPU** | AMD Ryzen 9 3900X (12 cores, 24 threads) | Excelente para procesamiento paralelo y offloading |
| **RAM** | 128GB DDR4 | Permite ejecutar modelos masivos con offloading de GPU |
| **GPU** | NVIDIA RTX 5060 Ti 16GB VRAM | Suficiente para modelos hasta ~27B en Q4, parcial para 70B |
| **Storage** | SSD recomendado | Crítico para carga rápida de modelos |

#### Capacidad de modelos

**Modelos que caben completamente en VRAM (16GB):**
- Hasta 14B con Q4 (~8-9GB)
- Hasta 12B con Q8 (~13GB)
- Hasta 27B con Q4 (~16GB, justo)

**Modelos con offloading a RAM:**
- 70B con Q4 (~40GB): Parcialmente en GPU, resto en RAM
- Velocidad reducida pero funcional gracias a 128GB RAM

---

## 2. Preparación del Sistema

### 2.1 Actualización del sistema

Antes de comenzar, es fundamental tener el sistema completamente actualizado:

```bash
# Actualizar lista de repositorios
sudo apt update

# Actualizar todos los paquetes instalados
sudo apt upgrade -y

# Actualizar paquetes del sistema (incluyendo kernel si necesario)
sudo apt full-upgrade -y

# Limpiar paquetes obsoletos
sudo apt autoremove -y
sudo apt autoclean
```

**Nota importante:** Si se actualiza el kernel, es necesario reiniciar:

```bash
# Verificar si hay actualizaciones pendientes de kernel
uname -r  # Ver versión actual
ls /boot | grep vmlinuz  # Ver versiones disponibles

# Si hay nueva versión de kernel, reiniciar
sudo reboot
```

### 2.2 Verificación de hardware

Antes de proceder, verificamos que el sistema detecta correctamente el hardware:

#### Verificar CPU

```bash
# Información detallada del CPU
lscpu

# Información específica de rendimiento
lscpu | grep -E "Model name|CPU\(s\)|Thread|Socket|Core|MHz"
```

**Salida esperada para Ryzen 9 3900X:**
```
Model name:            AMD Ryzen 9 3900X 12-Core Processor
CPU(s):                24
Thread(s) per core:    2
Core(s) per socket:    12
Socket(s):             1
```

#### Verificar RAM

```bash
# Ver memoria total y disponible
free -h

# Ver detalles de módulos RAM
sudo dmidecode --type memory | grep -E "Size|Speed|Type:"
```

**Salida esperada:**
```
              total        used        free      shared  buff/cache   available
Mem:          126Gi       2.5Gi       120Gi       100Mi       3.5Gi       123Gi
Swap:         8.0Gi          0B       8.0Gi
```

#### Verificar GPU

```bash
# Listar dispositivos PCI (aún sin drivers puede no mostrar detalles)
lspci | grep -i nvidia

# Verificar ranura PCIe
lspci -v | grep -i nvidia
```

**Salida esperada:**
```
09:00.0 VGA compatible controller: NVIDIA Corporation Device XXXX (rev a1)
```

#### Verificar almacenamiento

```bash
# Ver espacio disponible
df -h

# Ver información de discos
lsblk

# Velocidad de disco (requiere permisos root)
sudo hdparm -Tt /dev/sda  # Reemplazar con tu disco
```

**Recomendación:** Los modelos ocupan espacio significativo:
- Modelo 8B Q4: ~4-5GB
- Modelo 14B Q4: ~8-9GB
- Modelo 70B Q4: ~40GB

Se recomienda tener al menos **100GB libres** para trabajar cómodamente.

### 2.3 Herramientas esenciales

Instalación de herramientas necesarias para el desarrollo y diagnóstico:

```bash
# Compiladores y herramientas de desarrollo
sudo apt install -y build-essential

# Herramientas de red y descarga
sudo apt install -y curl wget git

# Herramientas de monitoreo
sudo apt install -y htop iotop nethogs

# Headers del kernel (necesarios para compilar módulos)
sudo apt install -y linux-headers-$(uname -r)

# Herramientas de gestión de paquetes
sudo apt install -y software-properties-common apt-transport-https ca-certificates gnupg lsb-release

# Python y herramientas asociadas
sudo apt install -y python3 python3-pip python3-venv

# Utilidades adicionales
sudo apt install -y tmux screen nano vim
```

---

## 3. Instalación de Drivers NVIDIA

### 3.1 Identificación de GPU

Ubuntu 24.04 incluye una herramienta para detectar automáticamente los drivers necesarios:

```bash
# Detectar drivers recomendados
ubuntu-drivers devices
```

**Salida esperada:**
```
== /sys/devices/pci0000:00/0000:00:03.1/0000:09:00.0 ==
modalias : pci:v000010DEd00002684sv00001462sd00005141bc03sc00i00
vendor   : NVIDIA Corporation
model    : GA104 [GeForce RTX 5060 Ti]
driver   : nvidia-driver-550 - distro non-free recommended
driver   : nvidia-driver-545 - distro non-free
driver   : nvidia-driver-535 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

### 3.2 Limpieza de instalaciones previas

Si has tenido instalaciones previas de drivers NVIDIA, es recomendable limpiarlas:

```bash
# Detener servidor gráfico (solo si estás en servidor o puedes acceder por SSH)
sudo systemctl stop gdm3  # o lightdm, dependiendo de tu display manager

# Remover drivers antiguos
sudo apt remove --purge '^nvidia-.*' -y
sudo apt remove --purge '^libnvidia-.*' -y
sudo apt remove --purge '^cuda-.*' -y

# Limpiar paquetes huérfanos
sudo apt autoremove -y
sudo apt autoclean

# Eliminar configuraciones residuales
sudo rm -rf /etc/modprobe.d/nvidia*
sudo rm -rf /etc/modprobe.d/blacklist-nvidia*

# Actualizar initramfs
sudo update-initramfs -u

# Reiniciar
sudo reboot
```

### 3.3 Instalación de drivers

Existen dos métodos principales para instalar drivers NVIDIA en Ubuntu:

#### Método A: Repositorios de Ubuntu (RECOMENDADO)

Este método es el más estable y mejor integrado con el sistema:

```bash
# Opción 1: Instalación automática del driver recomendado
sudo ubuntu-drivers autoinstall

# Opción 2: Instalación manual de versión específica
# Ver versiones disponibles
apt-cache search nvidia-driver

# Instalar versión específica (recomendado: 550 o superior)
sudo apt install nvidia-driver-550 -y

# Verificar que se instaló correctamente
dpkg -l | grep nvidia-driver
```

#### Método B: Repositorio PPA (drivers más recientes)

Si necesitas la versión más reciente:

```bash
# Agregar PPA de drivers gráficos
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Instalar última versión estable
sudo apt install nvidia-driver-565 -y
```

#### Configuración post-instalación

```bash
# Asegurar que nouveau está deshabilitado (driver de código abierto)
echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# Actualizar initramfs
sudo update-initramfs -u

# IMPORTANTE: Reiniciar para cargar nuevos drivers
sudo reboot
```

### 3.4 Verificación y testing

Después del reinicio, verificar que los drivers funcionan correctamente:

```bash
# Comando básico de verificación
nvidia-smi
```

**Salida esperada:**
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07    CUDA Version: 12.4     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off | 00000000:09:00.0  On |                  N/A |
|  0%   42C    P8              15W / 220W |    256MiB / 16384MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

**Comandos adicionales de verificación:**

```bash
# Ver versión del driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Ver información detallada de la GPU
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,power.draw --format=csv

# Ver procesos usando la GPU
nvidia-smi pmon

# Monitor en tiempo real (actualización cada segundo)
watch -n 1 nvidia-smi
```

---

## 4. Instalación de CUDA Toolkit

CUDA (Compute Unified Device Architecture) es la plataforma de computación paralela de NVIDIA necesaria para ejecutar cargas de trabajo de ML/AI.

### 4.1 Instalación de CUDA 12.x

Para Ollama, se recomienda CUDA 12.x por compatibilidad y rendimiento:

```bash
# Descargar el keyring de repositorios CUDA para Ubuntu 24.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

# Instalar el keyring
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Actualizar lista de paquetes
sudo apt update

# Instalar CUDA Toolkit 12.6 (o la versión más reciente)
sudo apt install cuda-toolkit-12-6 -y

# El paquete cuda-toolkit incluye:
# - Compilador CUDA (nvcc)
# - Bibliotecas de desarrollo
# - Herramientas de profiling
# - Samples y documentación
```

**Verificar instalación:**

```bash
# Listar paquetes CUDA instalados
dpkg -l | grep cuda

# Ver versión de CUDA instalada
ls -l /usr/local/cuda*
```

### 4.2 Configuración de variables de entorno

Es crucial configurar correctamente las variables de entorno para que las aplicaciones encuentren CUDA:

```bash
# Editar archivo de configuración del shell
nano ~/.bashrc

# Agregar al final del archivo las siguientes líneas:
# ============== CUDA CONFIGURATION ==============
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Para mejor compatibilidad
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME

# Guardar archivo (Ctrl+O, Enter, Ctrl+X en nano)

# Recargar configuración
source ~/.bashrc

# Verificar que las variables están configuradas
echo $CUDA_HOME
echo $PATH | grep cuda
```

**Verificar que nvcc funciona:**

```bash
# Ver versión del compilador CUDA
nvcc --version
```

**Salida esperada:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Mon_Sep_16_21:23:40_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

### 4.3 Instalación de cuDNN

cuDNN (CUDA Deep Neural Network library) es una biblioteca de primitivas GPU-aceleradas para redes neuronales:

```bash
# Instalar cuDNN 9 para CUDA 12
sudo apt install libcudnn9-cuda-12 -y
sudo apt install libcudnn9-dev-cuda-12 -y

# Verificar instalación
dpkg -l | grep cudnn
```

**Nota:** cuDNN es opcional para Ollama pero mejora el rendimiento de ciertos modelos.

---

## 5. Instalación de Ollama

### 5.1 Instalación básica

Ollama proporciona un script de instalación oficial que detecta automáticamente el sistema:

```bash
# Descargar e instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# El script realizará:
# 1. Descarga del binario de Ollama
# 2. Instalación en /usr/local/bin/ollama
# 3. Creación del servicio systemd
# 4. Creación del usuario 'ollama'
# 5. Configuración de permisos
```

**Verificar instalación:**

```bash
# Ver versión instalada
ollama --version

# Debería mostrar algo como:
# ollama version is 0.x.x
```

### 5.2 Configuración del servicio

Ollama se instala como un servicio systemd que se ejecuta en segundo plano:

```bash
# Ver estado del servicio
sudo systemctl status ollama

# Iniciar servicio (si no está iniciado)
sudo systemctl start ollama

# Habilitar inicio automático al arrancar
sudo systemctl enable ollama

# Ver logs del servicio
journalctl -u ollama -f

# Reiniciar servicio (útil después de cambios de configuración)
sudo systemctl restart ollama
```

### 5.3 Verificación de GPU

Ollama debe detectar automáticamente la GPU NVIDIA si los drivers y CUDA están correctamente instalados:

```bash
# Ver logs iniciales del servicio
sudo journalctl -u ollama -n 50 --no-pager | grep -i -E "gpu|cuda|nvidia"
```

**Deberías ver líneas como:**
```
Oct 23 10:30:15 hostname ollama[12345]: time=2025-10-23T10:30:15.123+00:00 level=INFO source=gpu.go:123 msg="looking for compatible GPUs"
Oct 23 10:30:15 hostname ollama[12345]: time=2025-10-23T10:30:15.456+00:00 level=INFO source=gpu.go:234 msg="discovered GPU" id=0 library=cuda compute=8.9 driver=12.4 name="NVIDIA GeForce RTX 5060 Ti" total="16.0 GiB"
```

**Si no detecta la GPU:**

```bash
# Verificar que nvidia-smi funciona
nvidia-smi

# Verificar que CUDA está en el PATH
which nvcc

# Verificar variables de entorno
env | grep -i cuda

# Reinstalar Ollama con CUDA explícito
curl -fsSL https://ollama.com/install.sh | OLLAMA_CUDA=cuda_v12 sh
```

---

## 6. Configuración Avanzada de Ollama

### 6.1 Variables de entorno del servicio

Para aprovechar al máximo tu hardware, es necesario configurar variables de entorno específicas del servicio:

```bash
# Crear directorio de configuración personalizada
sudo mkdir -p /etc/systemd/system/ollama.service.d

# Crear archivo de configuración
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

**Contenido del archivo `override.conf`:**

```ini
[Service]
# === CONFIGURACIÓN DE RED ===
# Permitir conexiones desde cualquier IP (útil para acceso remoto)
Environment="OLLAMA_HOST=0.0.0.0:11434"

# Permitir CORS desde cualquier origen
Environment="OLLAMA_ORIGINS=*"

# === CONFIGURACIÓN DE RENDIMIENTO ===
# Número de requests paralelos (aprovecha CPU de 24 threads)
Environment="OLLAMA_NUM_PARALLEL=4"

# Número máximo de modelos cargados simultáneamente
Environment="OLLAMA_MAX_LOADED_MODELS=3"

# Tamaño de cola de requests
Environment="OLLAMA_MAX_QUEUE=512"

# === CONFIGURACIÓN DE MEMORIA ===
# Directorio de modelos (personalizar si es necesario)
Environment="OLLAMA_MODELS=/home/TU_USUARIO/.ollama/models"

# Tiempo que modelos permanecen en memoria tras uso
Environment="OLLAMA_KEEP_ALIVE=5m"

# Usar Flash Attention para reducir uso de VRAM
Environment="OLLAMA_FLASH_ATTENTION=1"

# === CONFIGURACIÓN DE GPU ===
# Usar GPU 0 (si tienes múltiples GPUs)
Environment="CUDA_VISIBLE_DEVICES=0"

# Biblioteca CUDA a usar
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"

# VRAM máxima a usar (15360 MB = 15 GB, dejar 1GB para sistema)
Environment="OLLAMA_MAX_VRAM=15360"

# Overhead de GPU en MB
Environment="OLLAMA_GPU_OVERHEAD=512"

# === LOGGING ===
# Nivel de logging (debug, info, warn, error)
Environment="OLLAMA_DEBUG=0"
```

**⚠️ IMPORTANTE:** 
- Reemplazar `TU_USUARIO` con tu nombre de usuario real de Ubuntu
- Los valores son optimizados para RTX 5060 Ti 16GB

**Aplicar la configuración:**

```bash
# Recargar configuración de systemd
sudo systemctl daemon-reload

# Reiniciar servicio Ollama
sudo systemctl restart ollama

# Verificar que las variables se aplicaron correctamente
sudo systemctl show ollama | grep Environment
```

### 6.2 Configuración de usuario

Además de la configuración del servicio, puedes configurar variables para tu usuario:

```bash
# Editar archivo de configuración del shell
nano ~/.bashrc

# Agregar al final:
# ============== OLLAMA USER CONFIGURATION ==============

# URL del servidor Ollama
export OLLAMA_HOST="http://localhost:11434"

# Directorio de modelos
export OLLAMA_MODELS="$HOME/.ollama/models"

# Número de GPUs a usar
export OLLAMA_NUM_GPU=1

# Cargar máximo número de capas en GPU (999 = todas las posibles)
export OLLAMA_GPU_LAYERS=999

# Tamaño de contexto por defecto (tokens)
# 4096 es estándar, puedes aumentar a 8192 o 16384 si tienes RAM suficiente
export OLLAMA_CONTEXT_SIZE=4096

# Flash Attention (reduce uso de VRAM significativamente)
export OLLAMA_FLASH_ATTENTION=1

# Nivel de paralelización de procesamiento (threads CPU)
export OLLAMA_NUM_THREAD=12  # Usar mitad de threads del CPU

# ======================================================

# Guardar archivo (Ctrl+O, Enter, Ctrl+X en nano)

# Recargar configuración
source ~/.bashrc
```

### 6.3 Optimización de memoria

#### Configuración de tamaño de contexto dinámico

Diferentes modelos y tareas requieren diferentes tamaños de contexto:

```bash
# Para conversaciones normales (4K contexto)
export OLLAMA_CONTEXT_SIZE=4096

# Para documentos largos (8K contexto)
export OLLAMA_CONTEXT_SIZE=8192

# Para análisis exhaustivo (16K-32K contexto)
# ADVERTENCIA: Aumenta significativamente el uso de memoria
export OLLAMA_CONTEXT_SIZE=16384
```

**Cálculo de memoria necesaria:**
- Contexto 4K: ~2GB adicionales de VRAM
- Contexto 8K: ~4GB adicionales de VRAM
- Contexto 16K: ~8GB adicionales de VRAM

#### Configuración de batch size

```bash
# Batch size para inferencia (mayor = más rápido pero más memoria)
export OLLAMA_NUM_BATCH=512

# Para VRAM limitada, reducir:
export OLLAMA_NUM_BATCH=256
```

---

## 7. Selección y Descarga de Modelos

### 7.1 Modelos recomendados para RAG

#### A. Modelos de Embedding

Los modelos de embedding convierten texto en vectores numéricos para búsqueda semántica.

**Recomendación principal: mxbai-embed-large**

```bash
ollama pull mxbai-embed-large
```

**Características:**
- Dimensiones: 1024
- Tamaño: ~670MB
- Contexto: 512 tokens
- Ventaja: Alto rendimiento, ampliamente usado en producción
- Precisión: >90% en benchmarks RAG

**Alternativas:**

```bash
# Para soporte multilingüe superior
ollama pull qwen3-embedding:8b

# Para máxima velocidad
ollama pull nomic-embed-text

# Para alta precisión (mayor tamaño)
ollama pull bge-large
```

**Comparación de modelos de embedding:**

| Modelo | Parámetros | Tamaño | Dimensiones | Velocidad | Precisión |
|--------|------------|--------|-------------|-----------|-----------|
| mxbai-embed-large | 334M | 670MB | 1024 | Alta | 92%+ |
| nomic-embed-text | 137M | 274MB | 768 | Muy alta | 88%+ |
| qwen3-embedding:8b | 8B | 4.7GB | 1024 | Media | 94%+ |
| bge-large | 335M | 670MB | 1024 | Alta | 91%+ |

#### B. Modelos de Generación para RAG

**Recomendación para velocidad: llama3.2:8b**

```bash
# Versión optimizada con cuantización Q4_K_M
ollama pull llama3.2:8b-instruct-q4_K_M
```

**Características:**
- Tamaño: ~4.7GB
- VRAM necesaria: ~5-6GB
- Velocidad en RTX 5060 Ti: 40-60 tokens/segundo
- Calidad: Excelente para la mayoría de tareas RAG

**Recomendación para calidad máxima: llama3.3:70b**

```bash
# Requiere offloading a RAM
ollama pull llama3.3:70b-instruct-q4_K_M
```

**Características:**
- Tamaño: ~40GB
- VRAM: Parcial (~10-12GB en GPU, resto en RAM)
- Velocidad: 8-15 tokens/segundo
- Calidad: Estado del arte, comparable a GPT-4

**Balance perfecto: qwen3:14b**

```bash
ollama pull qwen3:14b-instruct-q4_K_M
```

**Características:**
- Tamaño: ~8.5GB
- VRAM: ~9GB (cabe perfectamente)
- Velocidad: 30-40 tokens/segundo
- Calidad: Excelente balance, multilingüe superior

### 7.2 Modelos para conversación

#### A. Uso diario y análisis general

**qwen3:14b - Recomendación principal**

```bash
ollama pull qwen3:14b-instruct-q4_K_M
```

**Ventajas:**
- Excelente razonamiento general
- Soporte multilingüe superior (100+ idiomas)
- Buena velocidad
- Cabe completamente en VRAM

#### B. Razonamiento avanzado

**deepseek-r1:8b - Para análisis complejos**

```bash
ollama pull deepseek-r1:8b-q4_K_M
```

**Ventajas:**
- Especializado en razonamiento paso a paso
- Excelente para matemáticas y lógica
- Capacidad de auto-verificación
- Menor tamaño que otros modelos de razonamiento

**deepseek-r1:32b - Para máxima capacidad de razonamiento**

```bash
ollama pull deepseek-r1:32b-q4_K_M
```

**Características:**
- Tamaño: ~19GB
- Requiere offloading parcial
- Velocidad: 15-25 tokens/segundo
- Razonamiento cercano a GPT-4

#### C. Modelos especializados

**Para código:**

```bash
# DeepSeek Coder - Especialista en programación
ollama pull deepseek-coder:6.7b-instruct-q4_K_M

# Para análisis de código más avanzado
ollama pull deepseek-coder:33b-instruct-q4_K_M
```

**Para visión (multimodal):**

```bash
# Llama 3.2 Vision - Análisis de imágenes
ollama pull llama3.2-vision:11b-instruct-q4_K_M

# Alternativa más pequeña
ollama pull llava:7b
```

### 7.3 Script de descarga automatizada

Crear un script para descargar todos los modelos recomendados:

```bash
#!/bin/bash

# =============================================================================
# Script de instalación de modelos Ollama
# Optimizado para: RTX 5060 Ti 16GB + 128GB RAM
# =============================================================================

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Instalación de Modelos Ollama"
echo "=========================================="
echo "Hardware: RTX 5060 Ti 16GB + 128GB RAM"
echo "Sistema: Ubuntu 24.04"
echo ""

# Función para descargar modelos con feedback
download_model() {
    local model=$1
    local description=$2
    
    echo -e "\n${BLUE}Descargando: $model${NC}"
    echo -e "${YELLOW}Descripción: $description${NC}"
    
    if ollama pull "$model"; then
        echo -e "${GREEN}✓ $model descargado correctamente${NC}"
        
        # Mostrar información del modelo
        local size=$(ollama list | grep "$model" | awk '{print $2}')
        echo -e "${GREEN}  Tamaño: $size${NC}"
    else
        echo -e "${RED}✗ Error descargando $model${NC}"
        return 1
    fi
}

# =============================================================================
# MODELOS DE EMBEDDING
# =============================================================================
echo -e "\n${BLUE}=== MODELOS DE EMBEDDING ===${NC}"
echo "Los modelos de embedding se usan para convertir texto en vectores para RAG"

download_model "mxbai-embed-large" "Embedding principal - Alta precisión (1024 dims)"
download_model "nomic-embed-text" "Embedding rápido - Alternativa ligera (768 dims)"

# =============================================================================
# MODELOS DE GENERACIÓN (Elige uno según prioridad)
# =============================================================================
echo -e "\n${BLUE}=== MODELOS DE GENERACIÓN PARA RAG ===${NC}"

echo -e "\n${YELLOW}Selecciona el modelo de generación:${NC}"
echo "1) llama3.2:8b (RECOMENDADO) - Rápido, cabe en VRAM (~5GB)"
echo "2) qwen3:14b - Balance calidad/velocidad (~9GB)"
echo "3) llama3.3:70b - Máxima calidad, requiere RAM (~40GB)"
echo "4) Instalar todos"
echo "5) Saltar"

read -p "Opción (1-5): " gen_choice

case $gen_choice in
    1)
        download_model "llama3.2:8b-instruct-q4_K_M" "Generación rápida - 40-60 tok/s"
        ;;
    2)
        download_model "qwen3:14b-instruct-q4_K_M" "Balance perfecto - 30-40 tok/s"
        ;;
    3)
        download_model "llama3.3:70b-instruct-q4_K_M" "Máxima calidad - 8-15 tok/s"
        ;;
    4)
        download_model "llama3.2:8b-instruct-q4_K_M" "Generación rápida"
        download_model "qwen3:14b-instruct-q4_K_M" "Balance perfecto"
        download_model "llama3.3:70b-instruct-q4_K_M" "Máxima calidad"
        ;;
    5)
        echo "Saltando modelos de generación..."
        ;;
    *)
        echo "Opción inválida, instalando modelo recomendado..."
        download_model "llama3.2:8b-instruct-q4_K_M" "Generación rápida"
        ;;
esac

# =============================================================================
# MODELOS PARA CONVERSACIÓN
# =============================================================================
echo -e "\n${BLUE}=== MODELOS PARA CONVERSACIÓN ===${NC}"

download_model "qwen3:14b-instruct-q4_K_M" "Conversación general - Multilingüe superior"

# =============================================================================
# MODELOS ESPECIALIZADOS
# =============================================================================
echo -e "\n${BLUE}=== MODELOS ESPECIALIZADOS (OPCIONAL) ===${NC}"

read -p "¿Instalar modelo de razonamiento avanzado? (s/n): " reasoning_choice
if [[ $reasoning_choice =~ ^[Ss]$ ]]; then
    download_model "deepseek-r1:8b-q4_K_M" "Razonamiento y análisis complejo"
fi

read -p "¿Instalar modelo para programación? (s/n): " code_choice
if [[ $code_choice =~ ^[Ss]$ ]]; then
    download_model "deepseek-coder:6.7b-instruct-q4_K_M" "Generación y análisis de código"
fi

read -p "¿Instalar modelo multimodal (visión)? (s/n): " vision_choice
if [[ $vision_choice =~ ^[Ss]$ ]]; then
    download_model "llama3.2-vision:11b-instruct-q4_K_M" "Análisis de imágenes y documentos"
fi

# =============================================================================
# RESUMEN
# =============================================================================
echo -e "\n${GREEN}=========================================="
echo "Instalación completada!"
echo "==========================================${NC}"

echo -e "\n${BLUE}Modelos instalados:${NC}"
ollama list

echo -e "\n${BLUE}Espacio usado por modelos:${NC}"
du -sh ~/.ollama/models

echo -e "\n${YELLOW}Comandos útiles:${NC}"
echo "  ollama list                     - Ver modelos instalados"
echo "  ollama run <modelo>             - Ejecutar modelo interactivo"
echo "  ollama rm <modelo>              - Eliminar modelo"
echo "  ollama show <modelo>            - Ver detalles del modelo"

echo -e "\n${GREEN}¡Sistema listo para usar!${NC}"
```

**Guardar y ejecutar:**

```bash
# Guardar script
nano ~/install_ollama_models.sh

# Copiar el contenido anterior

# Hacer ejecutable
chmod +x ~/install_ollama_models.sh

# Ejecutar
~/install_ollama_models.sh
```

### 7.4 Gestión de modelos

#### Comandos básicos

```bash
# Listar todos los modelos instalados
ollama list

# Ver detalles de un modelo específico
ollama show llama3.2:8b-instruct-q4_K_M

# Copiar/duplicar modelo
ollama cp llama3.2:8b-instruct-q4_K_M my-custom-model

# Eliminar modelo
ollama rm nombre-modelo

# Actualizar modelo (descarga nueva versión si existe)
ollama pull llama3.2:8b-instruct-q4_K_M
```

#### Ver espacio usado

```bash
# Espacio total usado por modelos
du -sh ~/.ollama/models

# Espacio por modelo
du -h ~/.ollama/models/blobs | sort -h

# Detalles del directorio de modelos
tree ~/.ollama/models -L 2
```

#### Crear modelos personalizados con Modelfile

Los Modelfiles permiten personalizar parámetros de modelos existentes:

```bash
# Crear Modelfile personalizado
nano ~/Modelfile.llama-optimized
```

**Contenido del Modelfile:**

```dockerfile
# Modelo base
FROM llama3.2:8b-instruct-q4_K_M

# Parámetros de generación
PARAMETER temperature 0.7          # Creatividad (0.0-2.0)
PARAMETER top_p 0.9                # Nucleus sampling
PARAMETER top_k 40                 # Top-K sampling
PARAMETER repeat_penalty 1.1      # Penalización por repetición
PARAMETER num_ctx 8192            # Tamaño de contexto
PARAMETER num_gpu 999             # Capas en GPU (999 = todas)
PARAMETER num_thread 12           # Threads CPU

# System prompt personalizado
SYSTEM """
Eres un asistente útil, preciso y conciso especializado en tecnología.
Respondes en español de manera clara y estructurada.
Prefieres explicaciones técnicas pero accesibles.
Si no sabes algo, lo admites directamente.
"""

# Template de mensajes (opcional)
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
"""
```

**Crear modelo desde Modelfile:**

```bash
# Crear modelo personalizado
ollama create llama-optimized -f ~/Modelfile.llama-optimized

# Probar el modelo
ollama run llama-optimized "Explica qué es Docker"

# Listar para verificar
ollama list | grep llama-optimized
```

#### Exportar e importar modelos

```bash
# Exportar modelo a archivo (útil para backup)
ollama save llama3.2:8b-instruct-q4_K_M -o ~/llama-backup.tar

# Importar modelo desde archivo
ollama load ~/llama-backup.tar
```

---

## 8. Optimizaciones del Sistema

### 8.1 Configuración de memoria

#### Optimizar parámetros de memoria del kernel

```bash
# Editar configuración del kernel
sudo nano /etc/sysctl.conf

# Agregar al final:
# === Optimizaciones para LLMs ===

# Reducir swappiness (menos uso de swap, más uso de RAM)
# Con 128GB RAM, queremos evitar swap casi por completo
vm.swappiness=10

# Permitir overcommit de memoria
# Útil para modelos grandes que requieren offloading
vm.overcommit_memory=1

# Aumentar memoria compartida máxima (64GB)
kernel.shmmax=68719476736
kernel.shmall=4294967296

# Aumentar límite de archivos abiertos
fs.file-max=2097152

# Optimizar cache de páginas
vm.dirty_ratio=40
vm.dirty_background_ratio=10

# Guardar archivo (Ctrl+O, Enter, Ctrl+X)

# Aplicar cambios sin reiniciar
sudo sysctl -p

# Verificar que se aplicaron
sysctl vm.swappiness
sysctl vm.overcommit_memory
```

#### Configurar límites de recursos por usuario

```bash
# Editar límites de sistema
sudo nano /etc/security/limits.conf

# Agregar al final:
# Límites para usuario que ejecuta Ollama
*    soft    nofile    1048576
*    hard    nofile    1048576
*    soft    memlock   unlimited
*    hard    memlock   unlimited
*    soft    stack     unlimited
*    hard    stack     unlimited

# Guardar archivo

# Verificar límites actuales
ulimit -a

# Para aplicar, cerrar sesión y volver a iniciar
```

#### Configurar Transparent Huge Pages (THP)

```bash
# Ver configuración actual
cat /sys/kernel/mm/transparent_hugepage/enabled

# Desactivar THP (puede mejorar rendimiento para cargas ML)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Hacer permanente
sudo nano /etc/rc.local

# Agregar:
#!/bin/bash
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
exit 0

# Hacer ejecutable
sudo chmod +x /etc/rc.local
```

### 8.2 Governor de CPU

El governor de CPU controla cómo se ajusta la frecuencia del procesador. Para máximo rendimiento:

```bash
# Instalar herramientas de gestión de CPU
sudo apt install cpufrequtils linux-tools-common linux-tools-$(uname -r) -y

# Ver governor actual
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Ver governors disponibles
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors

# Establecer governor en modo performance (para todos los cores)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done

# Verificar
cpufreq-info | grep "current policy"
```

**Hacer permanente:**

```bash
# Crear archivo de configuración
sudo nano /etc/default/cpufrequtils

# Agregar:
GOVERNOR="performance"

# Aplicar y habilitar servicio
sudo systemctl restart cpufrequtils
sudo systemctl enable cpufrequtils

# Alternativa: Usar systemd
sudo systemctl disable ondemand
```

**Verificar frecuencias del CPU:**

```bash
# Ver frecuencias en tiempo real
watch -n 1 "grep MHz /proc/cpuinfo"

# O con cpupower (más detallado)
sudo cpupower frequency-info
```

### 8.3 Configuración de GPU

#### Modo de persistencia

El modo de persistencia mantiene el driver NVIDIA cargado incluso cuando no hay aplicaciones usándolo, reduciendo latencia:

```bash
# Activar modo de persistencia
sudo nvidia-smi -pm 1

# Verificar
nvidia-smi | grep "Persistence"
```

#### Configurar límites de potencia

```bash
# Ver límite de potencia actual
nvidia-smi -q -d POWER

# Establecer al máximo (ajustar según tu modelo específico)
# RTX 5060 Ti típicamente tiene TDP de 220W
sudo nvidia-smi -pl 220

# Ver clock speeds disponibles
nvidia-smi -q -d SUPPORTED_CLOCKS

# Establecer clocks fijos (opcional, puede generar más calor)
# GPU clock a máximo (ajustar según tu GPU)
sudo nvidia-smi -lgc 2100

# Memory clock a máximo
sudo nvidia-smi -lmc 9501
```

#### Hacer configuración permanente

```bash
# Crear servicio systemd para configuración GPU
sudo nano /etc/systemd/system/nvidia-performance.service
```

**Contenido del servicio:**

```ini
[Unit]
Description=NVIDIA GPU Performance Settings
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -pl 220
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
# Activar y habilitar servicio
sudo systemctl daemon-reload
sudo systemctl enable nvidia-performance.service
sudo systemctl start nvidia-performance.service

# Verificar
sudo systemctl status nvidia-performance.service
```

### 8.4 Herramientas de monitoreo

#### Instalar nvtop (monitor GPU avanzado)

```bash
# Instalar nvtop desde repositorios
sudo apt install nvtop -y

# Ejecutar
nvtop

# Interfaz similar a htop pero para GPU
# Muestra: uso GPU, VRAM, temperatura, procesos
```

#### Instalar btop (monitor de sistema completo)

```bash
# Instalar btop
sudo snap install btop

# O compilar desde source para última versión
git clone https://github.com/aristocratos/btop.git
cd btop
make
sudo make install

# Ejecutar
btop

# Muestra: CPU, RAM, disco, red, procesos en interfaz bonita
```

#### Instalar glances (monitor con API web)

```bash
# Instalar glances
sudo apt install glances -y

# Ejecutar en modo terminal
glances

# Ejecutar con servidor web (accesible en http://localhost:61208)
glances -w

# Con autenticación
glances -w --password
```

#### Crear aliases útiles

```bash
# Editar .bashrc
nano ~/.bashrc

# Agregar al final:
# === Aliases de monitoreo ===
alias gpu='watch -n 1 nvidia-smi'
alias gpumon='nvtop'
alias sysmon='btop'
alias gputemp='nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
alias ollamalogs='journalctl -u ollama -f'
alias ollamastatus='systemctl status ollama'

# Guardar y recargar
source ~/.bashrc

# Ahora puedes usar:
# gpu          - Ver estado de GPU cada segundo
# gpumon       - Abrir nvtop
# sysmon       - Abrir btop
# gputemp      - Ver temperatura GPU
# gpumem       - Ver memoria GPU usada/total
```

---

## 9. Testing y Benchmarking

### 9.1 Tests básicos

#### Script de verificación completa

```bash
# Crear script de test
nano ~/test_ollama_complete.sh
```

**Contenido:**

```bash
#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================"
echo "Test Completo de Ollama"
echo "================================================"

# Test 1: Servicio Ollama
echo -e "\n${BLUE}=== TEST 1: Estado del Servicio ===${NC}"
if systemctl is-active --quiet ollama; then
    echo -e "${GREEN}✓ Servicio Ollama está corriendo${NC}"
else
    echo -e "${RED}✗ Servicio Ollama NO está corriendo${NC}"
    echo "Iniciando servicio..."
    sudo systemctl start ollama
    sleep 3
fi

# Test 2: Detección de GPU
echo -e "\n${BLUE}=== TEST 2: Detección de GPU ===${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader

# Test 3: CUDA
echo -e "\n${BLUE}=== TEST 3: CUDA ===${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo -e "${GREEN}✓ CUDA instalado correctamente${NC}"
else
    echo -e "${RED}✗ CUDA no encontrado en PATH${NC}"
fi

# Test 4: Modelos instalados
echo -e "\n${BLUE}=== TEST 4: Modelos Instalados ===${NC}"
ollama list

# Test 5: Conexión a API
echo -e "\n${BLUE}=== TEST 5: API de Ollama ===${NC}"
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓ API accesible${NC}"
else
    echo -e "${RED}✗ API no accesible${NC}"
fi

# Test 6: Test de inferencia simple
echo -e "\n${BLUE}=== TEST 6: Test de Inferencia ===${NC}"
MODEL="llama3.2:8b-instruct-q4_K_M"

if ollama list | grep -q "$MODEL"; then
    echo "Ejecutando test con $MODEL..."
    echo "Prompt: 'Responde solo con: Hola'"
    
    time_start=$(date +%s.%N)
    response=$(ollama run $MODEL "Responde solo con: Hola" 2>&1)
    time_end=$(date +%s.%N)
    time_diff=$(echo "$time_end - $time_start" | bc)
    
    echo "Respuesta: $response"
    echo "Tiempo: ${time_diff}s"
    echo -e "${GREEN}✓ Test de inferencia completado${NC}"
else
    echo -e "${YELLOW}Modelo $MODEL no encontrado${NC}"
fi

# Test 7: Uso de recursos durante inferencia
echo -e "\n${BLUE}=== TEST 7: Monitoreo de Recursos ===${NC}"
echo "Ejecutando nvidia-smi en segundo plano..."
echo "Ejecuta un modelo en otra terminal para ver uso de recursos"
echo -e "${YELLOW}Presiona Ctrl+C para salir${NC}"
watch -n 1 nvidia-smi

echo -e "\n${GREEN}================================================"
echo "Tests completados"
echo "================================================${NC}"
```

```bash
# Hacer ejecutable
chmod +x ~/test_ollama_complete.sh

# Ejecutar
~/test_ollama_complete.sh
```

### 9.2 Benchmark de modelos

Script para comparar rendimiento de diferentes modelos:

```bash
nano ~/benchmark_ollama_models.sh
```

**Contenido:**

```bash
#!/bin/bash

# Configuración
PROMPTS=(
    "Explica qué es machine learning en 50 palabras"
    "Escribe un programa Python que calcule fibonacci"
    "Analiza las ventajas de usar Docker"
)

MODELS=(
    "llama3.2:8b-instruct-q4_K_M"
    "qwen3:14b-instruct-q4_K_M"
)

OUTPUT_DIR="$HOME/ollama_benchmarks"
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Benchmark de Modelos Ollama"
echo "================================================"
echo "Fecha: $(date)"
echo "Modelos: ${MODELS[*]}"
echo "Output: $OUTPUT_DIR"
echo ""

# Función de benchmark
benchmark_model() {
    local model=$1
    local prompt=$2
    local output_file="$OUTPUT_DIR/${model//[:\/]/_}_benchmark.txt"
    
    echo "Testing: $model"
    echo "Prompt: $prompt"
    
    # Calentar modelo (primera ejecución es más lenta)
    ollama run $model "test" >/dev/null 2>&1
    
    # Benchmark real
    echo "=== Benchmark: $model ===" >> "$output_file"
    echo "Prompt: $prompt" >> "$output_file"
    echo "Timestamp: $(date)" >> "$output_file"
    
    # Medir tiempo y capturar estadísticas
    /usr/bin/time -v ollama run $model "$prompt" 2>&1 | tee -a "$output_file"
    
    echo "" >> "$output_file"
    echo "---" >> "$output_file"
    echo ""
}

# Ejecutar benchmarks
for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo ""
        echo "=========================================="
        echo "Benchmarking: $model"
        echo "=========================================="
        
        for prompt in "${PROMPTS[@]}"; do
            benchmark_model "$model" "$prompt"
            sleep 2  # Pausa entre tests
        done
    else
        echo "ADVERTENCIA: Modelo $model no encontrado"
    fi
done

# Generar resumen
echo ""
echo "================================================"
echo "Benchmark completado"
echo "================================================"
echo "Resultados guardados en: $OUTPUT_DIR"
echo ""
echo "Archivos generados:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "Para ver resultados:"
echo "  cat $OUTPUT_DIR/*_benchmark.txt"
```

```bash
chmod +x ~/benchmark_ollama_models.sh
~/benchmark_ollama_models.sh
```

### 9.3 Test de RAG completo

Script Python para probar pipeline RAG completo:

```bash
# Crear entorno virtual
python3 -m venv ~/ollama-env
source ~/ollama-env/bin/activate

# Instalar dependencias
pip install ollama chromadb langchain sentence-transformers

# Crear script de test
nano ~/test_rag_complete.py
```

**Contenido:**

```python
#!/usr/bin/env python3
"""
Test completo de RAG con Ollama
Prueba: Embedding → Vector DB → Retrieval → Generation
"""

import ollama
import chromadb
import time
from typing import List, Dict

# Configuración
EMBEDDING_MODEL = "mxbai-embed-large"
GENERATION_MODEL = "llama3.2:8b-instruct-q4_K_M"

# Colores para output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def print_section(title: str):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{title}")
    print(f"{'='*60}{Colors.END}")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.YELLOW}→ {msg}{Colors.END}")

# Documentos de prueba sobre tecnología
documents = [
    "Python es un lenguaje de programación interpretado de alto nivel. Es conocido por su sintaxis clara y legible.",
    "Machine Learning es un subconjunto de la inteligencia artificial que permite a los sistemas aprender y mejorar automáticamente.",
    "Docker es una plataforma de contenedores que permite empaquetar aplicaciones con todas sus dependencias.",
    "Kubernetes es un sistema de orquestación de contenedores que automatiza el despliegue y la gestión de aplicaciones.",
    "Ubuntu 24.04 LTS es la última versión de soporte largo plazo de Ubuntu, lanzada en abril de 2024.",
    "NVIDIA RTX 5060 Ti es una tarjeta gráfica con 16GB de VRAM, ideal para machine learning y gaming.",
    "Ollama es una herramienta que permite ejecutar Large Language Models localmente en tu propia máquina.",
    "RAG (Retrieval-Augmented Generation) combina búsqueda de información con generación de texto.",
    "CUDA es la plataforma de computación paralela de NVIDIA para procesamiento GPU.",
    "Transformers son arquitecturas de redes neuronales que revolucionaron el procesamiento de lenguaje natural."
]

queries = [
    "¿Qué es Python?",
    "¿Cómo funciona Machine Learning?",
    "¿Para qué sirve Docker?",
    "¿Qué GPU recomendarías para ML?"
]

def main():
    print_section("Test Completo de RAG con Ollama")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Generation Model: {GENERATION_MODEL}")
    print(f"Documentos: {len(documents)}")
    print(f"Queries: {len(queries)}")
    
    # 1. Crear base de datos vectorial
    print_section("1. Creando Base de Datos Vectorial")
    
    client = chromadb.Client()
    collection = client.create_collection(
        name="tech_docs",
        metadata={"description": "Documentos técnicos de prueba"}
    )
    
    print_info(f"Colección 'tech_docs' creada")
    
    # 2. Generar embeddings e indexar documentos
    print_section("2. Generando Embeddings")
    
    start_time = time.time()
    
    for i, doc in enumerate(documents):
        # Generar embedding
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=doc
        )
        embedding = response["embeddings"]
        
        # Agregar a colección
        collection.add(
            ids=[str(i)],
            embeddings=embedding,
            documents=[doc],
            metadatas=[{"index": i, "length": len(doc)}]
        )
        
        print_success(f"Documento {i+1}/{len(documents)} indexado")
    
    embed_time = time.time() - start_time
    print_info(f"Tiempo total de embedding: {embed_time:.2f}s")
    print_info(f"Promedio por documento: {embed_time/len(documents):.2f}s")
    
    # 3. Realizar búsquedas y generar respuestas
    print_section("3. Pruebas de RAG")
    
    for query in queries:
        print(f"\n{Colors.YELLOW}{'─'*60}")
        print(f"Query: {query}")
        print(f"{'─'*60}{Colors.END}")
        
        # 3.1 Búsqueda semántica
        print_info("Buscando documentos relevantes...")
        
        query_start = time.time()
        query_response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_embedding = query_response["embeddings"]
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        retrieval_time = time.time() - query_start
        
        print_success(f"Búsqueda completada en {retrieval_time:.3f}s")
        print("\nDocumentos relevantes:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"  {i+1}. {doc[:80]}...")
        
        # 3.2 Generación con contexto
        print_info("\nGenerando respuesta...")
        
        context = "\n".join(results['documents'][0])
        prompt = f"""Basándote en el siguiente contexto, responde la pregunta de manera concisa y precisa.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""
        
        gen_start = time.time()
        response = ollama.generate(
            model=GENERATION_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 150
            }
        )
        gen_time = time.time() - gen_start
        
        print(f"\n{Colors.GREEN}Respuesta:")
        print(response['response'])
        print(f"{Colors.END}")
        
        # Estadísticas
        if 'eval_count' in response and 'eval_duration' in response:
            tokens = response['eval_count']
            duration_s = response['eval_duration'] / 1e9
            tokens_per_sec = tokens / duration_s
            
            print_info(f"Generación: {gen_time:.2f}s")
            print_info(f"Tokens: {tokens}")
            print_info(f"Velocidad: {tokens_per_sec:.2f} tokens/s")
    
    # 4. Resumen final
    print_section("4. Resumen del Test")
    
    print(f"✓ {len(documents)} documentos indexados correctamente")
    print(f"✓ {len(queries)} queries procesadas exitosamente")
    print(f"✓ Tiempo promedio de embedding: {embed_time/len(documents):.2f}s")
    print(f"✓ Sistema RAG funcionando correctamente")
    
    print(f"\n{Colors.GREEN}{'='*60}")
    print("Test completado exitosamente!")
    print(f"{'='*60}{Colors.END}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrumpido por usuario")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        raise
```

```bash
chmod +x ~/test_rag_complete.py
python3 ~/test_rag_complete.py
```

---

## 10. Troubleshooting

### 10.1 Problemas comunes

#### Problema 1: Ollama no detecta la GPU

**Síntomas:**
- Modelos muy lentos
- No se ve uso de GPU en nvidia-smi
- Logs muestran "no GPU detected"

**Solución:**

```bash
# 1. Verificar que drivers funcionan
nvidia-smi

# 2. Verificar CUDA
nvcc --version

# 3. Ver logs de Ollama
sudo journalctl -u ollama -n 100 --no-pager | grep -i "gpu\|cuda\|nvidia"

# 4. Verificar variables de entorno del servicio
sudo systemctl show ollama | grep -i cuda

# 5. Si no detecta GPU, reinstalar con CUDA explícito
sudo systemctl stop ollama
curl -fsSL https://ollama.com/install.sh | OLLAMA_CUDA=cuda_v12 sh
sudo systemctl start ollama

# 6. Verificar detección en logs
sudo journalctl -u ollama -f
# Ejecutar un modelo en otra terminal para ver logs
```

#### Problema 2: "Out of Memory" en GPU

**Síntomas:**
- Error al cargar modelo
- Modelo se carga pero falla al generar
- nvidia-smi muestra VRAM llena

**Soluciones:**

```bash
# Opción 1: Reducir contexto
export OLLAMA_CONTEXT_SIZE=2048  # En lugar de 4096 o más

# Opción 2: Usar modelo con mayor cuantización
ollama pull llama3.2:8b-instruct-q4_0  # En lugar de q4_K_M

# Opción 3: Liberar memoria
# Cerrar otros procesos que usan GPU
nvidia-smi
# Identificar PIDs y cerrar si es necesario
kill -9 <PID>

# Opción 4: Reiniciar servicio Ollama
sudo systemctl restart ollama

# Opción 5: Configurar offloading parcial
# Editar override.conf para usar menos VRAM
sudo nano /etc/systemd/system/ollama.service.d/override.conf
# Cambiar: OLLAMA_MAX_VRAM=12288  # Usar solo 12GB
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

#### Problema 3: Modelos extremadamente lentos

**Síntomas:**
- <5 tokens/segundo en modelo 8B
- CPU al 100%, GPU al 0%
- Respuestas tardan minutos

**Soluciones:**

```bash
# 1. Verificar que usa GPU
nvidia-smi -l 1
# En otra terminal, ejecutar modelo y ver si hay actividad GPU

# 2. Verificar configuración de capas GPU
export OLLAMA_GPU_LAYERS=999

# 3. Verificar que Flash Attention está activado
export OLLAMA_FLASH_ATTENTION=1

# 4. Verificar que el modelo está completamente descargado
ollama pull <modelo>

# 5. Verificar temperatura GPU (puede hacer throttling)
nvidia-smi --query-gpu=temperature.gpu --format=csv
# Si está >80°C, mejorar ventilación

# 6. Verificar governor CPU
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Cambiar a performance si no lo está
```

#### Problema 4: Servicio Ollama no inicia

**Síntomas:**
- `systemctl status ollama` muestra "failed"
- Error al ejecutar comandos ollama
- Puerto 11434 no responde

**Soluciones:**

```bash
# 1. Ver error específico
sudo journalctl -u ollama -xe

# 2. Verificar que puerto no está ocupado
sudo netstat -tlnp | grep 11434
# Si está ocupado, identificar proceso y cerrar

# 3. Verificar permisos del directorio de modelos
ls -la ~/.ollama/
sudo chown -R $USER:$USER ~/.ollama/

# 4. Verificar configuración del servicio
sudo systemctl cat ollama

# 5. Resetear configuración
sudo rm /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 6. Reinstalar Ollama
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
curl -fsSL https://ollama.com/install.sh | sh
```

#### Problema 5: Error de conexión a API

**Síntomas:**
- `curl http://localhost:11434` falla
- Aplicaciones no pueden conectar a Ollama
- Timeout en requests

**Soluciones:**

```bash
# 1. Verificar que servicio está corriendo
sudo systemctl status ollama

# 2. Verificar puerto y host
sudo systemctl show ollama | grep OLLAMA_HOST

# 3. Probar conexión local
curl http://localhost:11434/api/tags

# 4. Verificar firewall
sudo ufw status
# Si está bloqueando, abrir puerto
sudo ufw allow 11434/tcp

# 5. Probar con IP específica
OLLAMA_HOST="http://127.0.0.1:11434" ollama list

# 6. Revisar logs
sudo journalctl -u ollama -f
```

### 10.2 Comandos de diagnóstico

Script completo para diagnóstico del sistema:

```bash
nano ~/diagnose_ollama_complete.sh
```

**Contenido:**

```bash
#!/bin/bash

OUTPUT_FILE="$HOME/ollama_diagnostico_$(date +%Y%m%d_%H%M%S).txt"

echo "Iniciando diagnóstico completo..."
echo "Guardando en: $OUTPUT_FILE"

{
    echo "=========================================="
    echo "DIAGNÓSTICO COMPLETO DE OLLAMA"
    echo "=========================================="
    echo "Fecha: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    
    echo "=========================================="
    echo "1. SISTEMA OPERATIVO"
    echo "=========================================="
    lsb_release -a
    uname -a
    echo ""
    
    echo "=========================================="
    echo "2. HARDWARE - CPU"
    echo "=========================================="
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Socket|Core|MHz"
    echo ""
    
    echo "=========================================="
    echo "3. HARDWARE - RAM"
    echo "=========================================="
    free -h
    echo ""
    
    echo "=========================================="
    echo "4. HARDWARE - GPU"
    echo "=========================================="
    nvidia-smi
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw --format=csv
    echo ""
    
    echo "=========================================="
    echo "5. DRIVERS NVIDIA"
    echo "=========================================="
    dpkg -l | grep nvidia-driver
    echo ""
    modinfo nvidia | grep -E "version|filename"
    echo ""
    
    echo "=========================================="
    echo "6. CUDA"
    echo "=========================================="
    if command -v nvcc &> /dev/null; then
        nvcc --version
    else
        echo "CUDA no encontrado en PATH"
    fi
    echo ""
    ls -la /usr/local/cuda*
    echo ""
    
    echo "=========================================="
    echo "7. VARIABLES DE ENTORNO"
    echo "=========================================="
    env | grep -E "CUDA|OLLAMA|PATH" | sort
    echo ""
    
    echo "=========================================="
    echo "8. OLLAMA - VERSION Y ESTADO"
    echo "=========================================="
    ollama --version
    echo ""
    sudo systemctl status ollama --no-pager
    echo ""
    
    echo "=========================================="
    echo "9. OLLAMA - CONFIGURACIÓN"
    echo "=========================================="
    sudo systemctl show ollama | grep Environment
    echo ""
    
    echo "=========================================="
    echo "10. OLLAMA - MODELOS INSTALADOS"
    echo "=========================================="
    ollama list
    echo ""
    
    echo "=========================================="
    echo "11. ESPACIO EN DISCO"
    echo "=========================================="
    df -h
    echo ""
    echo "Espacio usado por modelos Ollama:"
    du -sh ~/.ollama/models 2>/dev/null || echo "Directorio no encontrado"
    echo ""
    
    echo "=========================================="
    echo "12. PROCESOS USANDO GPU"
    echo "=========================================="
    nvidia-smi pmon -c 1
    echo ""
    
    echo "=========================================="
    echo "13. PUERTOS EN ESCUCHA"
    echo "=========================================="
    sudo netstat -tlnp | grep -E "11434|ollama"
    echo ""
    
    echo "=========================================="
    echo "14. ÚLTIMOS LOGS DE OLLAMA"
    echo "=========================================="
    sudo journalctl -u ollama -n 50 --no-pager
    echo ""
    
    echo "=========================================="
    echo "15. CONFIGURACIÓN DE MEMORIA"
    echo "=========================================="
    sysctl vm.swappiness
    sysctl vm.overcommit_memory
    cat /sys/kernel/mm/transparent_hugepage/enabled
    echo ""
    
    echo "=========================================="
    echo "16. GOVERNOR DE CPU"
    echo "=========================================="
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | head -n 1
    echo ""
    
    echo "=========================================="
    echo "DIAGNÓSTICO COMPLETADO"
    echo "=========================================="
    
} | tee "$OUTPUT_FILE"

echo ""
echo "Diagnóstico guardado en: $OUTPUT_FILE"
echo ""
echo "Para compartir el diagnóstico:"
echo "  cat $OUTPUT_FILE"
```

```bash
chmod +x ~/diagnose_ollama_complete.sh
~/diagnose_ollama_complete.sh
```

### 10.3 Reset completo

Si todo falla y necesitas empezar desde cero:

```bash
# ADVERTENCIA: Esto eliminará todos los modelos descargados

# 1. Detener servicio
sudo systemctl stop ollama

# 2. Backup de modelos (opcional, si quieres conservarlos)
tar -czf ~/ollama_models_backup_$(date +%Y%m%d).tar.gz ~/.ollama/models

# 3. Eliminar Ollama completamente
sudo rm -rf /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama
sudo rm -rf /etc/systemd/system/ollama.service
sudo rm -rf /etc/systemd/system/ollama.service.d
sudo rm -rf ~/.ollama

# 4. Eliminar usuario ollama (si existe)
sudo userdel -r ollama 2>/dev/null

# 5. Recargar systemd
sudo systemctl daemon-reload

# 6. Reinstalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 7. Reconfigurar según sección 5 y 6 de este manual

# 8. Restaurar modelos (si hiciste backup)
mkdir -p ~/.ollama
tar -xzf ~/ollama_models_backup_*.tar.gz -C ~/

# 9. Verificar instalación
ollama --version
sudo systemctl status ollama
ollama list
```

---

## 11. Scripts de Utilidad

### 11.1 Script de inicio rápido

```bash
nano ~/ollama_start.sh
```

**Contenido:**

```bash
#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

clear

echo -e "${BLUE}"
echo "╔════════════════════════════════════════╗"
echo "║        OLLAMA QUICK START              ║"
echo "╚════════════════════════════════════════╝"
echo -e "${NC}"

# Verificar estado del sistema
echo -e "${BLUE}Estado del Sistema:${NC}"
echo "─────────────────────────────────────────"

# GPU
echo -ne "GPU: "
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv,noheader

# Servicio Ollama
echo -ne "Ollama: "
if systemctl is-active --quiet ollama; then
    echo -e "${GREEN}✓ Corriendo${NC}"
else
    echo -e "${YELLOW}✗ Detenido - Iniciando...${NC}"
    sudo systemctl start ollama
    sleep 2
    echo -e "${GREEN}✓ Iniciado${NC}"
fi

# Modelos
echo -e "\n${BLUE}Modelos Disponibles:${NC}"
echo "─────────────────────────────────────────"
ollama list

# Comandos útiles
echo -e "\n${BLUE}Comandos Útiles:${NC}"
echo "─────────────────────────────────────────"
echo "  ollama run <modelo>        - Ejecutar modelo"
echo "  ollama list                - Ver modelos"
echo "  ollama pull <modelo>       - Descargar modelo"
echo "  ollama rm <modelo>         - Eliminar modelo"
echo ""
echo "  gpu                        - Ver estado GPU"
echo "  gpumon                     - Monitor GPU (nvtop)"
echo "  sysmon                     - Monitor sistema (btop)"
echo "  ollamalogs                 - Ver logs en vivo"
echo ""
echo -e "${GREEN}Sistema listo!${NC}"
```

```bash
chmod +x ~/ollama_start.sh

# Crear alias
echo "alias oll='~/ollama_start.sh'" >> ~/.bashrc
source ~/.bashrc

# Ahora puedes ejecutar simplemente:
oll
```

### 11.2 Script de limpieza

```bash
nano ~/ollama_cleanup.sh
```

**Contenido:**

```bash
#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗"
echo "║     OLLAMA CLEANUP UTILITY             ║"
echo "╚════════════════════════════════════════╝${NC}"

# Mostrar espacio actual
echo -e "\n${BLUE}Espacio Actual:${NC}"
echo "Total de modelos: $(du -sh ~/.ollama/models 2>/dev/null | cut -f1)"
echo ""

# Listar modelos
echo -e "${BLUE}Modelos Instalados:${NC}"
ollama list

echo ""
read -p "¿Deseas ver el espacio de cada modelo? (s/n): " show_sizes

if [[ $show_sizes =~ ^[Ss]$ ]]; then
    echo -e "\n${BLUE}Tamaño por modelo:${NC}"
    du -h ~/.ollama/models/blobs 2>/dev/null | sort -h | tail -n 20
fi

# Opciones de limpieza
echo -e "\n${YELLOW}Opciones de limpieza:${NC}"
echo "1) Eliminar modelo específico"
echo "2) Limpiar caché temporal"
echo "3) Limpiar logs antiguos"
echo "4) Mostrar modelos no usados recientemente"
echo "5) Hacer todo lo anterior"
echo "6) Salir"

read -p "Selecciona opción (1-6): " option

case $option in
    1)
        read -p "Nombre del modelo a eliminar: " model
        ollama rm "$model"
        echo -e "${GREEN}Modelo eliminado${NC}"
        ;;
    2)
        echo "Limpiando caché temporal..."
        rm -rf ~/.ollama/tmp/* 2>/dev/null
        rm -rf /tmp/ollama* 2>/dev/null
        echo -e "${GREEN}Caché limpiado${NC}"
        ;;
    3)
        echo "Limpiando logs antiguos de Ollama..."
        sudo journalctl --vacuum-time=7d --unit=ollama
        echo -e "${GREEN}Logs limpiados${NC}"
        ;;
    4)
        echo -e "\n${BLUE}Modelos por última modificación:${NC}"
        ls -lt ~/.ollama/models/manifests/registry.ollama.ai/library/ 2>/dev/null | head -n 10
        ;;
    5)
        echo "Ejecutando limpieza completa..."
        rm -rf ~/.ollama/tmp/* 2>/dev/null
        rm -rf /tmp/ollama* 2>/dev/null
        sudo journalctl --vacuum-time=7d --unit=ollama
        echo -e "${GREEN}Limpieza completa realizada${NC}"
        ;;
    6)
        echo "Saliendo..."
        exit 0
        ;;
    *)
        echo -e "${RED}Opción inválida${NC}"
        ;;
esac

# Mostrar espacio final
echo -e "\n${BLUE}Espacio Final:${NC}"
echo "Total de modelos: $(du -sh ~/.ollama/models 2>/dev/null | cut -f1)"
df -h ~/.ollama/models

echo -e "\n${GREEN}Limpieza completada!${NC}"
```

```bash
chmod +x ~/ollama_cleanup.sh
```

### 11.3 Script de actualización

```bash
nano ~/ollama_update.sh
```

**Contenido:**

```bash
#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗"
echo "║     OLLAMA UPDATE UTILITY              ║"
echo "╚════════════════════════════════════════╝${NC}"

# Actualizar Ollama
echo -e "\n${YELLOW}1. Actualizando Ollama...${NC}"
curl -fsSL https://ollama.com/install.sh | sh

# Reiniciar servicio
echo -e "\n${YELLOW}2. Reiniciando servicio...${NC}"
sudo systemctl restart ollama
sleep 3

# Verificar versión
echo -e "\n${GREEN}3. Nueva versión:${NC}"
ollama --version

# Actualizar modelos
echo -e "\n${YELLOW}4. ¿Actualizar modelos instalados?${NC}"
echo "Esto descargará las últimas versiones de todos los modelos"
read -p "Continuar? (s/n): " update_models

if [[ $update_models =~ ^[Ss]$ ]]; then
    echo -e "\n${BLUE}Actualizando modelos...${NC}"
    
    # Obtener lista de modelos
    models=$(ollama list | tail -n +2 | awk '{print $1}')
    
    for model in $models; do
        echo -e "\n${YELLOW}Actualizando: $model${NC}"
        ollama pull "$model"
    done
    
    echo -e "\n${GREEN}Modelos actualizados!${NC}"
fi

echo -e "\n${GREEN}Actualización completada!${NC}"
```

```bash
chmod +x ~/ollama_update.sh
```

---

## 12. Configuración de Red y API

### 12.1 Acceso remoto a Ollama

Si quieres acceder a Ollama desde otras máquinas en tu red:

```bash
# Ya configuramos OLLAMA_HOST=0.0.0.0:11434 en el servicio
# Esto permite conexiones desde cualquier IP

# Verificar configuración
sudo systemctl show ollama | grep OLLAMA_HOST

# Abrir puerto en firewall
sudo ufw allow 11434/tcp
sudo ufw reload

# Verificar que está escuchando en todas las interfaces
sudo netstat -tlnp | grep 11434
```

**Test desde otra máquina:**

```bash
# Reemplazar TU_IP con la IP de tu servidor Ubuntu
curl http://TU_IP:11434/api/tags

# Usar Ollama desde otra máquina
OLLAMA_HOST=http://TU_IP:11434 ollama list
```

### 12.2 Integración con aplicaciones

#### Python

```python
import ollama

# Configurar host (si no es localhost)
client = ollama.Client(host='http://localhost:11434')

# Generar respuesta
response = client.generate(
    model='llama3.2:8b-instruct-q4_K_M',
    prompt='Hola, ¿cómo estás?'
)
print(response['response'])

# Chat con contexto
messages = [
    {'role': 'user', 'content': '¿Cuál es la capital de Francia?'},
]
response = client.chat(
    model='llama3.2:8b-instruct-q4_K_M',
    messages=messages
)
print(response['message']['content'])
```

#### JavaScript/Node.js

```javascript
import ollama from 'ollama'

const response = await ollama.chat({
  model: 'llama3.2:8b-instruct-q4_K_M',
  messages: [{ role: 'user', content: 'Hola' }],
})

console.log(response.message.content)
```

#### cURL

```bash
# Generar texto
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:8b-instruct-q4_K_M",
  "prompt": "¿Por qué el cielo es azul?",
  "stream": false
}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:8b-instruct-q4_K_M",
  "messages": [
    {"role": "user", "content": "Hola"}
  ]
}'

# Embeddings
curl http://localhost:11434/api/embed -d '{
  "model": "mxbai-embed-large",
  "input": "Texto de ejemplo"
}'
```

---

## 13. Referencias y Recursos

### 13.1 Documentación oficial

- **Ollama GitHub**: https://github.com/ollama/ollama
- **Ollama Docs**: https://github.com/ollama/ollama/tree/main/docs
- **Biblioteca de modelos**: https://ollama.com/library
- **NVIDIA CUDA**: https://docs.nvidia.com/cuda/
- **Ubuntu Documentation**: https://help.ubuntu.com/

### 13.2 Modelos y benchmarks

#### Fuentes de información sobre modelos

1. **Hugging Face MTEB Leaderboard** (Embeddings)
   - URL: https://huggingface.co/spaces/mteb/leaderboard
   - Benchmarks de precisión de modelos de embedding

2. **Artificial Analysis** (LLMs)
   - URL: https://artificialanalysis.ai/
   - Comparativas de rendimiento de LLMs

3. **LMSys Chatbot Arena**
   - URL: https://chat.lmsys.org/
   - Rankings basados en votaciones humanas

#### Referencias citadas en esta guía

Basado en investigaciones de octubre 2025:

- **Modelos de embedding**: mxbai-embed-large alcanza >92% de precisión en benchmarks RAG con 1000 queries de prueba
- **Qwen3 Embedding 8B**: #1 en MTEB multilingüe (score 70.58, junio 2025)
- **Ollama ecosystem 2025**: Soporte para >100 modelos, cuantización INT4/INT2, contextos hasta 128K tokens
- **DeepSeek-R1**: Rendimiento cercano a GPT-4 en razonamiento, 95% más económico
- **Llama 3.3 70B**: Rendimiento competitivo con modelos mucho más grandes

### 13.3 Comunidad y soporte

- **Discord de Ollama**: https://discord.gg/ollama
- **Reddit r/LocalLLaMA**: https://reddit.com/r/LocalLLaMA
- **GitHub Discussions**: https://github.com/ollama/ollama/discussions

### 13.4 Herramientas complementarias

#### Interfaces gráficas para Ollama

1. **Open WebUI** (antes Ollama WebUI)
   ```bash
   docker run -d -p 3000:8080 \
     --add-host=host.docker.internal:host-gateway \
     -v open-webui:/app/backend/data \
     --name open-webui \
     --restart always \
     ghcr.io/open-webui/open-webui:main
   ```
   - Interfaz web completa estilo ChatGPT
   - Soporte para RAG, documentos, imágenes
   - URL: https://github.com/open-webui/open-webui

2. **Ollama UI**
   ```bash
   docker run -d -p 8080:8080 \
     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
     ghcr.io/ollama-ui/ollama-ui:latest
   ```
   - Interfaz simple y ligera
   - URL: https://github.com/ollama-ui/ollama-ui

3. **Chatbox**
   - Aplicación de escritorio multiplataforma
   - Descarga: https://chatboxai.app/

#### Herramientas de desarrollo

1. **LangChain**: Framework para aplicaciones LLM
   ```bash
   pip install langchain langchain-community
   ```

2. **LlamaIndex**: Framework para RAG
   ```bash
   pip install llama-index
   ```

3. **ChromaDB**: Base de datos vectorial
   ```bash
   pip install chromadb
   ```

4. **Weaviate**: Base de datos vectorial empresarial
   - URL: https://weaviate.io/

### 13.5 Libros y cursos recomendados

- **"Hands-On Large Language Models"** - Jay Alammar & Maarten Grootendorst
- **"Building LLMs for Production"** - Louis-François Bouchard
- **DeepLearning.AI Courses** - Cursos gratuitos sobre LLMs y RAG

---

## 14. Apéndices

### Apéndice A: Tabla de Compatibilidad de Hardware

#### Requisitos mínimos por tamaño de modelo

| Tamaño Modelo | RAM CPU | VRAM GPU | Almacenamiento | Velocidad Esperada |
|---------------|---------|----------|----------------|--------------------|
| 3B-4B Q4      | 4GB     | 3GB      | 2-3GB          | 50-80 tok/s        |
| 7B-8B Q4      | 8GB     | 5GB      | 4-5GB          | 30-60 tok/s        |
| 13B-14B Q4    | 16GB    | 9GB      | 8-9GB          | 20-40 tok/s        |
| 30B-34B Q4    | 32GB    | 18GB     | 18-20GB        | 10-20 tok/s        |
| 70B Q4        | 64GB    | 40GB*    | 38-42GB        | 5-15 tok/s         |

*Con offloading a RAM

#### Configuración recomendada según presupuesto

**Configuración Básica (800-1200€)**
- CPU: Ryzen 5 5600 / Intel i5-12400
- RAM: 32GB DDR4
- GPU: RTX 4060 Ti 16GB o RTX 3060 12GB
- Storage: 500GB NVMe SSD
- **Modelos recomendados**: Hasta 13B

**Configuración Media (1500-2500€)** ⭐ Tu configuración
- CPU: Ryzen 9 3900X / Ryzen 7 5800X
- RAM: 64-128GB DDR4
- GPU: RTX 5060 Ti 16GB / RTX 4070 Ti 16GB
- Storage: 1TB NVMe SSD
- **Modelos recomendados**: Hasta 70B con offloading

**Configuración Alta (3000-5000€)**
- CPU: Ryzen 9 7950X / Intel i9-13900K
- RAM: 128-192GB DDR5
- GPU: RTX 4090 24GB / RTX 6000 Ada 48GB
- Storage: 2TB NVMe SSD Gen4
- **Modelos recomendados**: Hasta 180B con offloading

**Configuración Profesional (8000+€)**
- CPU: Threadripper PRO / Xeon W
- RAM: 256GB+ DDR5 ECC
- GPU: 2x RTX 4090 / A6000 48GB
- Storage: 4TB+ NVMe RAID
- **Modelos recomendados**: Cualquier modelo, múltiples simultáneos

### Apéndice B: Glosario de Términos

**API (Application Programming Interface)**
Interfaz que permite a aplicaciones comunicarse con Ollama

**Batch Size**
Número de tokens procesados simultáneamente; mayor = más rápido pero más memoria

**GGUF (GPT-Generated Unified Format)**
Formato de archivo para modelos cuantizados, usado por Ollama

**Offloading**
Técnica de usar RAM cuando VRAM es insuficiente para el modelo completo

**Embedding**
Representación vectorial numérica de texto que captura significado semántico

**Context Window**
Cantidad de tokens que el modelo puede "recordar" en una conversación

**Fine-tuning**
Entrenamiento adicional de un modelo base para especializarlo

**Inference**
Proceso de usar un modelo para generar predicciones/respuestas

**KV Cache**
Caché de cálculos previos que acelera la generación de tokens subsecuentes

**Perplexity**
Métrica de calidad del modelo; menor = mejor

**Prompt Engineering**
Arte de formular preguntas/instrucciones para obtener mejores respuestas

**Quantization**
Reducción de precisión numérica para ahorrar memoria

**RAG (Retrieval-Augmented Generation)**
Técnica que combina búsqueda de información con generación de texto

**Temperature**
Parámetro que controla la aleatoriedad; 0 = determinista, 2 = muy creativo

**Token**
Unidad básica de texto procesada por el modelo (~0.75 palabras)

**Top-K / Top-P**
Parámetros de sampling que controlan diversidad de generación

**VRAM (Video RAM)**
Memoria de la GPU, crucial para cargar modelos

### Apéndice C: Tabla de Cuantización Detallada

| Cuantización | Bits | Tamaño Relativo | Pérdida Calidad | Velocidad | Uso Recomendado |
|--------------|------|-----------------|-----------------|-----------|-----------------|
| FP16         | 16   | 100%            | 0%              | Base      | Producción crítica |
| Q8_0         | 8    | 50%             | 1-2%            | +10%      | Alta calidad |
| Q6_K         | 6    | 38%             | 3-4%            | +15%      | Balance calidad |
| Q5_K_M       | 5    | 31%             | 4-6%            | +20%      | Recomendado |
| Q5_K_S       | 5    | 31%             | 5-7%            | +20%      | Más velocidad |
| Q4_K_M       | 4    | 25%             | 6-10%           | +30%      | **Óptimo** ⭐ |
| Q4_K_S       | 4    | 25%             | 8-12%           | +30%      | Más velocidad |
| Q4_0         | 4    | 25%             | 10-15%          | +35%      | Máxima velocidad |
| Q3_K_M       | 3    | 19%             | 15-20%          | +40%      | Solo emergencia |
| Q2_K         | 2    | 12%             | 25-35%          | +50%      | No recomendado |

**Leyenda:**
- **_M**: Medium - Balance entre tamaño y calidad
- **_S**: Small - Más comprimido, menor calidad
- **_L**: Large - Menos comprimido, mayor calidad
- **_K**: Variante optimizada con bloques de cuantización mixta

**Recomendación para tu hardware:** Q4_K_M ofrece el mejor balance

### Apéndice D: Comandos Rápidos de Referencia

#### Gestión de Ollama

```bash
# Servicio
sudo systemctl start ollama          # Iniciar
sudo systemctl stop ollama           # Detener
sudo systemctl restart ollama        # Reiniciar
sudo systemctl status ollama         # Ver estado
journalctl -u ollama -f              # Ver logs en vivo

# Modelos
ollama list                          # Listar modelos
ollama pull <modelo>                 # Descargar modelo
ollama rm <modelo>                   # Eliminar modelo
ollama show <modelo>                 # Ver detalles
ollama cp <origen> <destino>         # Copiar modelo

# Ejecución
ollama run <modelo>                  # Modo interactivo
ollama run <modelo> "pregunta"       # Una pregunta
```

#### Monitoreo

```bash
# GPU
nvidia-smi                           # Estado actual
nvidia-smi -l 1                      # Monitor continuo
nvidia-smi --query-gpu=...           # Query específico
nvtop                                # Monitor interactivo

# Sistema
htop                                 # Monitor CPU/RAM
btop                                 # Monitor completo
df -h                                # Espacio en disco
free -h                              # Memoria RAM
```

#### Diagnóstico

```bash
# Verificar instalación
nvidia-smi                           # Drivers GPU
nvcc --version                       # CUDA
ollama --version                     # Ollama

# Verificar configuración
env | grep -i ollama                 # Variables entorno
sudo systemctl show ollama           # Config servicio
curl http://localhost:11434/api/tags # API funcionando
```

### Apéndice E: Checklist de Instalación

Use esta checklist para verificar que todo está correctamente configurado:

**□ Preparación del Sistema**
- □ Ubuntu 24.04 actualizado
- □ Paquetes esenciales instalados
- □ Hardware verificado

**□ Drivers NVIDIA**
- □ Drivers instalados (nvidia-smi funciona)
- □ Versión 550+ instalada
- □ GPU detectada correctamente

**□ CUDA Toolkit**
- □ CUDA 12.x instalado
- □ nvcc funciona
- □ Variables de entorno configuradas ($CUDA_HOME, $PATH)

**□ Ollama**
- □ Ollama instalado (ollama --version)
- □ Servicio corriendo (systemctl status ollama)
- □ GPU detectada por Ollama (verificar logs)

**□ Configuración Avanzada**
- □ Variables del servicio configuradas (override.conf)
- □ Variables de usuario configuradas (.bashrc)
- □ Servicio reiniciado después de cambios

**□ Modelos**
- □ Al menos 1 modelo de embedding descargado
- □ Al menos 1 modelo de generación descargado
- □ Modelos listados correctamente (ollama list)

**□ Optimizaciones**
- □ Memoria del kernel optimizada (sysctl.conf)
- □ Governor CPU en modo performance
- □ GPU en modo persistencia
- □ Herramientas de monitoreo instaladas

**□ Testing**
- □ Test básico completado (test_ollama_complete.sh)
- □ Test de inferencia exitoso
- □ Velocidad aceptable (>30 tok/s en modelo 8B)

**□ Documentación**
- □ Scripts guardados en ~/ y ejecutables
- □ Aliases configurados en .bashrc
- □ Diagnóstico ejecutado y guardado

### Apéndice F: Preguntas Frecuentes (FAQ)

**P: ¿Cuánta VRAM necesito para modelo X?**
R: Regla general: Parámetros × 1.2 / nivel_cuantización
- Ejemplo 8B Q4: 8 × 1.2 / 4 = 2.4GB → ~5GB con contexto
- Ejemplo 70B Q4: 70 × 1.2 / 4 = 21GB → ~40GB con contexto

**P: ¿Por qué mi modelo es lento si tengo buena GPU?**
R: Verificar:
1. GPU realmente detectada (nvidia-smi durante ejecución)
2. OLLAMA_GPU_LAYERS=999 configurado
3. Modelo no es demasiado grande para VRAM (hace offloading)
4. Temperature GPU <80°C (no hace throttling)

**P: ¿Puedo usar modelos de Hugging Face en Ollama?**
R: Sí, pero requiere conversión a formato GGUF:
1. Descargar modelo de HF
2. Convertir con convert.py de llama.cpp
3. Cuantizar con llama.cpp
4. Crear Modelfile en Ollama
5. `ollama create nombre -f Modelfile`

**P: ¿Cómo mejoro la calidad de las respuestas?**
R: Ajustar parámetros:
- Temperature: 0.7-0.9 (menor = más determinista)
- Top_p: 0.9-0.95 (nucleus sampling)
- Repeat_penalty: 1.0-1.2 (penalización por repetir)
- Usar prompts más específicos y detallados

**P: ¿Puedo ejecutar múltiples modelos simultáneamente?**
R: Sí, configurar:
```bash
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_NUM_PARALLEL=4
```
Tu 128GB RAM y 16GB VRAM lo permiten.

**P: ¿Qué diferencia hay entre Q4_K_M y Q4_0?**
R: Q4_K_M usa cuantización mixta (bloques de diferentes precisiones) mientras Q4_0 usa cuantización uniforme. Q4_K_M es ~5-10% mejor en calidad con mismo tamaño.

**P: ¿Cómo actualizo Ollama sin perder modelos?**
R: Los modelos se guardan en ~/.ollama/models y no se eliminan al actualizar. Simplemente ejecutar:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**P: ¿Puedo usar Ollama con WSL2?**
R: Sí, pero con limitaciones:
- GPU passthrough requiere WSL2 con soporte CUDA
- Rendimiento ~10-20% menor que Linux nativo
- Recomendado: Ubuntu nativo para máximo rendimiento

**P: ¿Cómo exporto conversaciones/resultados?**
R: Ollama no guarda historial por defecto. Opciones:
1. Redirigir output: `ollama run modelo > output.txt`
2. Usar Open WebUI que sí guarda historial
3. Implementar logging en tu aplicación

**P: Mi GPU tiene 8GB VRAM, ¿qué modelos puedo usar?**
R: Modelos hasta ~7B Q4 comfortablemente. Para tu 16GB:
- Hasta 14B Q4 cómodamente
- Hasta 27B Q4 ajustado
- 70B Q4 con offloading

### Apéndice G: Plantillas de Prompts

#### Para RAG

```
Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta de manera precisa y concisa. Si la información no está en el contexto, indica que no puedes responder.

Contexto:
{context}

Pregunta: {question}

Respuesta:
```

#### Para análisis de código

```
Analiza el siguiente código {language} y proporciona:
1. Descripción breve de su función
2. Posibles errores o bugs
3. Sugerencias de mejora
4. Mejores prácticas no aplicadas

Código:
{language}
{code}


Análisis:
```

#### Para generación de documentación

```
Genera documentación técnica completa para el siguiente código. Incluye:
- Descripción general
- Parámetros y tipos
- Valores de retorno
- Ejemplos de uso
- Consideraciones especiales

Código:
{code}

Documentación:
```

#### Para resumen de documentos

```
Resume el siguiente texto en {n} puntos clave. Cada punto debe ser:
- Conciso (máximo 2 líneas)
- Específico y factual
- Autocontenido

Texto:
{text}

Resumen:
```

### Apéndice H: Troubleshooting por Síntoma

**Síntoma: Ollama muy lento (< 10 tok/s en 8B)**
→ Verificar uso de GPU con nvidia-smi
→ Revisar OLLAMA_GPU_LAYERS
→ Comprobar temperatura GPU
→ Verificar que modelo no excede VRAM

**Síntoma: Error "Connection refused"**
→ Verificar servicio: `systemctl status ollama`
→ Verificar puerto: `netstat -tlnp | grep 11434`
→ Revisar firewall: `ufw status`
→ Ver logs: `journalctl -u ollama -xe`

**Síntoma: "Out of memory" constante**
→ Reducir OLLAMA_CONTEXT_SIZE
→ Usar modelo con mayor cuantización (Q4_0 vs Q4_K_M)
→ Reducir OLLAMA_MAX_VRAM
→ Cerrar otros procesos GPU

**Síntoma: Respuestas de baja calidad**
→ Probar modelo más grande
→ Usar menor cuantización (Q5 vs Q4)
→ Ajustar temperature (0.7 óptimo)
→ Mejorar el prompt con más contexto

**Síntoma: GPU no detectada**
→ Verificar drivers: `nvidia-smi`
→ Verificar CUDA: `nvcc --version`
→ Revisar logs: `journalctl -u ollama | grep -i gpu`
→ Reinstalar con CUDA explícito

**Síntoma: Modelo se detiene a mitad de respuesta**
→ Aumentar num_predict (tokens máximos)
→ Verificar contexto no está lleno
→ Revisar repeat_penalty (no muy alto)
→ Comprobar estabilidad GPU (temperatura, potencia)

---

## Conclusión

Esta guía proporciona una configuración completa y optimizada de Ollama en Ubuntu 24.04, específicamente adaptada para hardware de alto rendimiento como el especificado (Ryzen 9 3900X, 128GB RAM, RTX 5060 Ti 16GB).

### Resumen de Configuración Óptima

Con esta configuración deberías lograr:

✅ **Modelos 8B-14B**: 30-60 tokens/segundo (completamente en VRAM)  
✅ **Modelos 27B**: 15-30 tokens/segundo (ajustado en VRAM)  
✅ **Modelos 70B**: 8-15 tokens/segundo (offloading a RAM)  
✅ **Embeddings**: <100ms por documento  
✅ **RAG completo**: <3s por query (búsqueda + generación)

### Próximos Pasos

1. **Experimentar con diferentes modelos** para encontrar el balance ideal para tus casos de uso
2. **Implementar tu pipeline RAG** personalizado usando los scripts de ejemplo
3. **Monitorear rendimiento** y ajustar parámetros según necesidades
4. **Explorar fine-tuning** para especializar modelos en tus dominios específicos
5. **Integrar con aplicaciones** usando las APIs proporcionadas

### Mantenimiento Recomendado

- **Semanal**: Verificar actualizaciones de Ollama (`~/ollama_update.sh`)
- **Mensual**: Limpiar modelos no usados (`~/ollama_cleanup.sh`)
- **Trimestral**: Actualizar drivers NVIDIA y CUDA
- **Anual**: Revisar nuevos modelos disponibles y mejores prácticas

### Soporte y Comunidad

Si encuentras problemas no cubiertos en esta guía:

1. Ejecutar diagnóstico completo: `~/diagnose_ollama_complete.sh`
2. Revisar logs detalladamente: `journalctl -u ollama -n 200`
3. Buscar en GitHub Issues: https://github.com/ollama/ollama/issues
4. Preguntar en Discord: https://discord.gg/ollama
5. Compartir diagnóstico en r/LocalLLaMA

---

**Fin del Manual**

**Autor:** Oscar Toledano Sole  
**Versión:** 1.0  
**Última actualización:** Octubre 2025  
**Licencia:** Creative Commons BY-SA 4.0

Para actualizaciones y correcciones, consultar la documentación oficial de Ollama.
