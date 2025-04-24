#Importing all three of the records by all three LLM 
"""
   # Extract, print and insert JSON data from each model's output.
    try:
        invoice_data_deepseek = JSONExtractor.extract_json(deepseek_output)
        print("\nExtracted JSON from DeepSeek:")
        print(json.dumps(invoice_data_deepseek, indent=4, ensure_ascii=False))
    except ValueError as e:
        print("Failed to extract JSON from DeepSeek output:", e)

    try:
        invoice_data_llama = JSONExtractor.extract_json(llama_output)
        print("\nExtracted JSON from Llama:")
        print(json.dumps(invoice_data_llama, indent=4, ensure_ascii=False))
    except ValueError as e:
        print("Failed to extract JSON from Llama output:", e)

    try:
        invoice_data_mistral = JSONExtractor.extract_json(mistral_output)
        print("\nExtracted JSON from Mistral:")
        print(json.dumps(invoice_data_mistral, indent=4, ensure_ascii=False))
    except ValueError as e:
        print("Failed to extract JSON from Mistral output:", e)

    # Inserting into database
    try:
        db = InvoiceDB()
        # Insert the invoice data from each model individually.
        invoice_id_deepseek = db.insert_invoice(invoice_data_deepseek)
        print("Inserted invoice from DeepSeek with ID:", invoice_id_deepseek)
        
        invoice_id_llama = db.insert_invoice(invoice_data_llama)
        print("Inserted invoice from Llama with ID:", invoice_id_llama)
        
        invoice_id_mistral = db.insert_invoice(invoice_data_mistral)
        print("Inserted invoice from Mistral with ID:", invoice_id_mistral)
        
        db.close()
    except Exception as e:
        print("Failed to insert invoice(s) into database:", e)
"""