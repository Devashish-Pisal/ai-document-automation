

def get_formated_prompt(ocr_data: str):
    prompt = f"""
    You are a highly precise financial document parsing system.
    Your task is to extract specific invoice data from raw OCR text. The OCR text may contain noise, formatting issues, or errors. You must interpret it carefully and return ONLY a strictly valid JSON object.

    1) CRITICAL OUTPUT RULES:
    - Output ONLY valid JSON. No explanations, no comments, no markdown, no extra text.
    - The JSON must be parseable by standard JSON parsers.
    - Do not include trailing commas.
    - Do not include any fields other than those specified.

    2) FIELDS TO EXTRACT:
    - company_name: string | null  
    - company_address: string | null  
    - invoice_date: string (YYYY-MM-DD) | null  
    - total_amount: number | null  

    4) EXTRACTION LOGIC:

    1. COMPANY NAME:
       - Typically appears at the top of the invoice.
       - Prefer the seller/vendor/issuer (NOT the customer).
       - Ignore labels like "Bill To", "Ship To".
    2. COMPANY ADDRESS:
       - Extract the full address of the issuing company.
       - Usually near the company name or header section.
       - Combine multi-line address into a single string.
    3. INVOICE DATE:
       - Prefer fields labeled "Invoice Date".
       - Ignore "Due Date", "Order Date", etc.
       - Normalize to YYYY-MM-DD format.
       - If ambiguous, choose the most likely invoice issue date.
    4. TOTAL AMOUNT:
       - STRICTLY select the final payable amount.
       - Prefer labels like:
         "Total", "Grand Total", "Amount Due", "Balance Due"
       - DO NOT select:
         Subtotal, Tax, VAT, Discount, Unit Price
       - Convert to a pure number:
         - Remove currency symbols (€,$,£,etc.)
         - Remove thousand separators
         - Use dot as decimal separator
       - Example: "€1,234.56" → 1234.56
    5. HANDLING OCR NOISE:
       - Correct obvious OCR mistakes (e.g., "T0tal" → "Total").
       - Be resilient to broken formatting, spacing, or line breaks.
    6. MISSING DATA:
       - If a field cannot be confidently found, return null.
    7. MULTIPLE CANDIDATES:
       - Choose the most semantically correct value based on context.
       - Prioritize labeled values over unlabeled numbers.

    5) INPUT OCR TEXT:
    ""
    {ocr_data}
    ""

    6) OUTPUT FORMAT (STRICT):
   {{
      "company_name": "string or null",
      "company_address": "string or null",
      "invoice_date": "YYYY-MM-DD or null",
      "total_amount": number or null
    }}"""
    return prompt