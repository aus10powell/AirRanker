def filter_listings(listings, filters):
    """
    Filter listings based on the provided filters.
    
    Args:
        listings (list): List of listing dictionaries
        filters (dict): Dictionary containing filter criteria
        
    Returns:
        list: Filtered listings
    """
    filtered = listings
    
    # Filter by keywords if provided
    if 'keywords' in filters and filters['keywords']:
        keywords = [k.lower() for k in filters['keywords']]
        filtered = [
            listing for listing in filtered
            if any(k in (listing.get('name', '') or '').lower() or 
                  k in (listing.get('description', '') or '').lower() 
                  for k in keywords)
        ]
    
    # Filter by location if provided
    if 'location' in filters and filters['location']:
        location = filters['location'].lower()
        filtered = [
            listing for listing in filtered
            if location in (listing.get('location', '') or '').lower()
        ]
    
    # Filter by price limit if provided
    if 'price_limit' in filters and filters['price_limit']:
        price_limit = float(filters['price_limit'])
        filtered = [
            listing for listing in filtered
            if listing.get('price', float('inf')) <= price_limit
        ]
    
    # Filter by amenities if provided
    if 'amenities' in filters and filters['amenities']:
        amenities = [a.lower() for a in filters['amenities']]
        filtered = [
            listing for listing in filtered
            if all(a in (listing.get('amenities', []) or []).lower() for a in amenities)
        ]
    
    return filtered 