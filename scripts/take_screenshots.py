"""
Script para tomar capturas de pantalla de la API usando Playwright.
Genera imagenes para documentacion del README.
"""
import asyncio
from playwright.async_api import async_playwright
import os

# Directorio de salida
OUTPUT_DIR = "docs/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_URL = "http://localhost:8000"

async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        # 1. Captura de la pagina principal
        print("1. Capturando pagina principal...")
        await page.goto(f"{API_URL}/")
        await page.screenshot(path=f"{OUTPUT_DIR}/01_api_welcome.png", full_page=True)

        # 2. Captura de la documentacion Swagger
        print("2. Capturando documentacion Swagger...")
        await page.goto(f"{API_URL}/docs")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(1)  # Esperar a que cargue completamente
        await page.screenshot(path=f"{OUTPUT_DIR}/02_api_docs.png", full_page=True)

        # 3. Expandir el endpoint POST /predict
        print("3. Expandiendo endpoint /predict...")
        try:
            # Click en el endpoint POST /predict
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
            await asyncio.sleep(1)  # Esperar respuesta
            await page.screenshot(path=f"{OUTPUT_DIR}/05_api_response.png", full_page=True)
        except Exception as e:
            print(f"   Advertencia: {e}")

        await browser.close()
        print(f"\nCapturas guardadas en: {OUTPUT_DIR}/")
        print("Archivos generados:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                print(f"  - {f}")

if __name__ == "__main__":
    asyncio.run(take_screenshots())