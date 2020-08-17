function filterByBrandWithTypes(array, brands){
    var filtered = [];
    var types = ["Food Delivery", "Banking & Finance", "Transport"];

    for(var i = 0; i < array.length; i++){
        var obj = array[i][1];
        // console.log(obj.toLowerCase());
        for(var j = 0; j < brands.length; j++){
            var brand = brands[j][0];
            // console.log(brand.toLowerCase());
            if(obj.toLowerCase().includes(brand.toLowerCase())){
                if(obj.toLowerCase().includes("food")){
                    array[i][2] = types[0];
                    filtered.push(array[i]);
                } else if(obj.toLowerCase().includes("bank")){
                    array[i][2] = types[1];
                    filtered.push(array[i]);
                } else{
                    array[i][2] = types[2];
                    filtered.push(array[i]);
                }
            }
        }
    }
    return filtered;
};