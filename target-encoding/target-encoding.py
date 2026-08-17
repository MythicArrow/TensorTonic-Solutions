def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    # Store sum and count for each category
    sum_dict = {}
    count_dict = {}
    
    # Accumulate values using a single loop
    for cat, tar in zip(categories, targets):
        sum_dict[cat] = sum_dict.get(cat, 0.0) + tar
        count_dict[cat] = count_dict.get(cat, 0) + 1
        
    # Calculate the mean for each category
    mean_dict = {cat: sum_dict[cat] / count_dict[cat] for cat in sum_dict}
    
    # Map the original categories to their mean values
    return [mean_dict[cat] for cat in categories]
