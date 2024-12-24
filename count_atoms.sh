#!/usr/bin/bash 
is_alpha_numeric() { #0 is capital, 1 is small, 2 is number
    local char=$1
    local flag=0
    if [[ $char =~ ^[A-Z]+$ ]];then flag=0
    elif [[ $char =~ ^[a-z]+$ ]];then flag=1
    elif [[ $char =~ ^[0-9]+$ ]];then flag=2
    else echo "Invalid character encountered in formula $1";exit 1
    fi
    return $flag
}

is_valid() { #Check the validity of the formula
    local flag=$1
    local last_encountered=$2
    if [ $last_encountered -eq -1 ];then #if the first values are number or small letter raise error
        if [ $flag -eq 2 ];then echo "Number encountered at the start of formula";exit 1;fi
        if [ $flag -eq 1 ];then echo "Small letter encountered at start of the formula";exit 1;fi
    fi
    if [ $last_encountered -eq 1  ] && [ $flag -eq 1 ];then #if two small letters are seen raise error
        echo "Two small letters enountered together";exit 1
    fi
    if [ $last_encountered -eq 2 ] && [ $flag -eq 1 ];then #if a small letter is enountered after a number
        echo "Small letter enountered after a number";exit 1;fi
    return 0
}

formula="$1X"
build_num=""
sum=0
length=${#formula}
last_encountered=-1
for ((i = 0; i < length; i++));do
    char="${formula:i:1}"
    is_alpha_numeric $char     
    flag=$?
    is_valid $flag $last_encountered
    if [ $flag -eq 2 ];then build_num="$build_num$char"
    elif [ $last_encountered -eq $flag ] || [ $last_encountered -eq $((flag + 1)) ];then 
        let "sum++"
        build_num=""
    elif [ $last_encountered -eq 2 ] && [ $flag -eq 0 ];then 
        sum=$(($sum+$build_num))
        build_num=""
    fi
    last_encountered=$flag
done
echo $sum

