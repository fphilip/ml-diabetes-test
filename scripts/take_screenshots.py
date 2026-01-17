"""
Script para tomar capturas de pantalla de la API y Frontend usando Playwright.
Genera imagenes para documentacion del README.
"""
import asyncio
from playwright.async_api import async_playwright
import os

# Directorio de salida
OUTPUT_DIR = "docs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"

# Casos preset del frontend
PRESET_CASES = {
    "P10": {"name": "Paciente 10", "expected": "Diabético", "color": "verde claro"},
    "P11": {"name": "Paciente 11", "expected": "Sano", "color": "rojo"},
    "P12": {"name": "Paciente 12", "expected": "Diabético", "color": "naranja"},
    "P13": {"name": "Paciente 13", "expected": "Sano", "color": "verde"},
}


async def take_api_screenshots(page):
    """Capturas de la API"""
    # 1. Captura de la pagina principal
    print("1. Capturando pagina principal API...")
    await page.goto(f"{API_URL}/")
    await page.screenshot(path=f"{OUTPUT_DIR}/01_api_welcome.png", full_page=True)

    # 2. Captura de la documentacion Swagger
    print("2. Capturando documentacion Swagger...")
    await page.goto(f"{API_URL}/docs")
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(1)
    await page.screenshot(path=f"{OUTPUT_DIR}/02_api_docs.png", full_page=True)

    # 3. Expandir el endpoint POST /predict
    print("3. Expandiendo endpoint /predict...")
    try:
        await page.click('text=POST')
        await asyncio.sleep(0.5)
        await page.screenshot(path=f"{OUTPUT_DIR}/03_api_predict_endpoint.png", full_page=True)
    except Exception as e:
        print(f"   Advertencia: {e}")

    # 4. Click en "Try it out"
    print("4. Activando 'Try it out'...")
    try:
        await page.click('text=Try it out')
        await asyncio.sleep(0.5)
        await page.screenshot(path=f"{OUTPUT_DIR}/04_api_try_it_out.png", full_page=True)
    except Exception as e:
        print(f"   Advertencia: {e}")

    # 5. Ejecutar prediccion
    print("5. Ejecutando prediccion...")
    try:
        await page.click('text=Execute')
        await asyncio.sleep(1)
        await page.screenshot(path=f"{OUTPUT_DIR}/05_api_response.png", full_page=True)
    except Exception as e:
        print(f"   Advertencia: {e}")


async def take_frontend_screenshots(page):
    """Capturas del Frontend con casos preset"""
    print("\n--- Capturas del Frontend ---")

    # 6. Captura del formulario vacio
    print("6. Capturando formulario vacio...")
    await page.goto(FRONTEND_URL)
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(3)  # Esperar que Streamlit cargue
    await page.screenshot(path=f"{OUTPUT_DIR}/06_frontend_empty.png", full_page=True)

    # Capturas de cada caso preset (2 por caso: formulario y resultados)
    img_num = 7
    for preset_key, preset_info in PRESET_CASES.items():
        print(f"{img_num}. Capturando caso {preset_info['name']} - Formulario...")
        try:
            # Recargar pagina para estado limpio
            await page.goto(FRONTEND_URL)
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(3)

            # Click en el boton del preset
            await page.click(f'button:has-text("{preset_key}")')
            await asyncio.sleep(2)

            # Captura del formulario con datos cargados
            await page.screenshot(
                path=f"{OUTPUT_DIR}/{img_num:02d}_frontend_{preset_key.lower()}_form.png",
                full_page=True
            )
            img_num += 1

            print(f"{img_num}. Capturando caso {preset_info['name']} - Resultados...")

            # Click en "Evaluar Paciente"
            await page.click('button:has-text("Evaluar Paciente")')

            # Esperar a que aparezca el grafico LIME
            await asyncio.sleep(12)  # LIME tarda en renderizar

            # Captura full page para ver todo incluyendo LIME
            await page.screenshot(
                path=f"{OUTPUT_DIR}/{img_num:02d}_frontend_{preset_key.lower()}_result.png",
                full_page=True
            )
            img_num += 1

        except Exception as e:
            print(f"   Advertencia en {preset_key}: {e}")
            img_num += 2  # Saltar numeros aunque falle


async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # Viewport muy alto para capturar todo el contenido
        page = await browser.new_page(viewport={"width": 1400, "height": 2000})

        # Capturas de la API
        await take_api_screenshots(page)

        # Capturas del Frontend
        await take_frontend_screenshots(page)

        await browser.close()

        print(f"\nCapturas guardadas en: {OUTPUT_DIR}/")
        print("Archivos generados:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                print(f"  - {f}")


if __name__ == "__main__":
    asyncio.run(take_screenshots())