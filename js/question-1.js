function filterByBrand(array, brands){
    var filtered = [];
    for(var i = 0; i < array.length; i++){
        var obj = array[i][1];
        // console.log(obj.toLowerCase());
        for(var j = 0; j < brands.length; j++){
            var brand = brands[j][0];
            // console.log(brand.toLowerCase());
            if(obj.toLowerCase().includes(brand.toLowerCase())){
                filtered.push(array[i]);
            }
        }
    }
    return filtered;
};