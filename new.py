from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Define the Pydantic model for the input data
class Item(BaseModel):
    name: str
    description: str
    price: float
    tax: float = None

# Initialize FastAPI app
app = FastAPI()

@app.post("/create_item")
def create_item(item: Item):
    # Example of using the status code for 'Created' (201)
    print(item.name)  # This will print the name of the item to the console
    return JSONResponse(
        content={"message": "Item created successfully", "item": item.dict(), "name": item.name},
        status_code=status.HTTP_201_CREATED  # HTTP 201 Created
    )
